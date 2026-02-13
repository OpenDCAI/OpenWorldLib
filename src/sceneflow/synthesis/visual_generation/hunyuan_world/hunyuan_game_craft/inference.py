import torch
from pathlib import Path
from loguru import logger
from .constants import PROMPT_TEMPLATE, PRECISION_TO_TYPE

from .modules import load_model
from .text_encoder import TextEncoder
import torch.distributed
from .modules.parallel_states import (
    initialize_sequence_parallel_state,
    get_sequence_parallel_state,
    nccl_info,
)
from .modules.fp8_optimization import convert_fp8_linear


from .....base_models.diffusion_model.video.hunyuan_video.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from .constants import VAE_PATH, PRECISION_TO_TYPE

def load_vae(vae_type,
             vae_precision=None,
             sample_size=None,
             vae_path=None,
             logger=None,
             device=None,
             model_base="tencent/Hunyuan-GameCraft-1.0"
             ):
    """
    Load and configure a Variational Autoencoder (VAE) model.
    
    This function handles loading 3D causal VAE models, including configuration,
    weight loading, precision setting, and device placement. It ensures the model
    is properly initialized for inference.

    Parameters:
        vae_type (str): Type identifier for the VAE, must follow '???-*' format for 3D VAEs
        vae_precision (str, optional): Desired precision type (e.g., 'fp16', 'fp32'). 
                                     Uses model's default if not specified.
        sample_size (tuple, optional): Input sample dimensions to override config defaults
        vae_path (str, optional): Path to VAE model files. Uses predefined path from
                                VAE_PATH constant if not specified.
        logger (logging.Logger, optional): Logger instance for progress/debug messages
        device (torch.device, optional): Target device to place the model (e.g., 'cuda' or 'cpu')

    Returns:
        tuple: Contains:
            - vae (AutoencoderKLCausal3D): Loaded and configured VAE model
            - vae_path (str): Actual path used to load the VAE
            - spatial_compression_ratio (int): Spatial dimension compression factor
            - time_compression_ratio (int): Temporal dimension compression factor

    Raises:
        ValueError: If vae_type does not follow the required 3D VAE format '???-*'
    """
    if vae_path is None:
        vae_path = f"{model_base}/{VAE_PATH[vae_type]}"
    vae_compress_spec, _, _ = vae_type.split("-")
    length = len(vae_compress_spec)
    # Process 3D VAE (valid format with 3-character compression spec)
    if length == 3:
        if logger is not None:
            logger.info(f"Loading 3D VAE model ({vae_type}) from: {vae_path}")
        config = AutoencoderKLCausal3D.load_config(vae_path)
        if sample_size:
            vae = AutoencoderKLCausal3D.from_config(config, sample_size=sample_size)
        else:
            vae = AutoencoderKLCausal3D.from_config(config)
        ckpt = torch.load(Path(vae_path) / "pytorch_model.pt", map_location=vae.device)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        vae_ckpt = {k.replace("vae.", ""): v for k, v in ckpt.items() if k.startswith("vae.")}
        vae.load_state_dict(vae_ckpt)

        spatial_compression_ratio = vae.config.spatial_compression_ratio
        time_compression_ratio = vae.config.time_compression_ratio
    else:
        raise ValueError(f"Invalid VAE model: {vae_type}. Must be 3D VAE in the format of '???-*'.")

    if vae_precision is not None:
        vae = vae.to(dtype=PRECISION_TO_TYPE[vae_precision])

    vae.requires_grad_(False)

    if logger is not None:
        logger.info(f"VAE to dtype: {vae.dtype}")

    if device is not None:
        vae = vae.to(device)

    # Ensure model is in evaluation mode (disables dropout/batch norm training behavior)
    # Note: Even with dropout rate 0, eval mode is recommended for consistent inference
    vae.eval()

    return vae, vae_path, spatial_compression_ratio, time_compression_ratio

class Inference(object):
    def __init__(self, 
                args,
                vae, 
                vae_kwargs, 
                text_encoder, 
                model, 
                text_encoder_2=None, 
                pipeline=None, 
                cpu_offload=False,
                device=None, 
                logger=None):
        self.vae = vae
        self.vae_kwargs = vae_kwargs
        
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        
        self.model = model
        self.pipeline = pipeline
        self.cpu_offload = cpu_offload
        
        self.args = args
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        if nccl_info.sp_size > 1:
            self.device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        
        self.logger = logger

    @classmethod
    def from_pretrained(cls, 
                        pretrained_model_path,
                        model_base,
                        args,
                        device=None,
                        **kwargs):
        """
        Initialize the Inference pipeline.

        Args:
            pretrained_model_path (str or pathlib.Path): The model path, 
            including t2v, text encoder and vae checkpoints.
            device (int): The device for inference. Default is 0.
            logger (logging.Logger): The logger for the inference pipeline. Default is None.
        """
        # ========================================================================
        logger.info(f"Got text-to-video model root path: {pretrained_model_path}")
        
        # ======================== Get the args path =============================
        
        # Set device and disable gradient
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)
        logger.info("Building model...")
        factor_kwargs = {'device': 'cpu' if args.cpu_offload else device, 'dtype': PRECISION_TO_TYPE[args.precision]}
        in_channels = args.latent_channels
        out_channels = args.latent_channels
        print("="*25, f"build model", "="*25)
        model = load_model(
            args,
            in_channels=in_channels,
            out_channels=out_channels,
            factor_kwargs=factor_kwargs
        )
        if args.cpu_offload:
            print(f'='*20, f'load transformer to cpu')
            model = model.to('cpu')
            torch.cuda.empty_cache()
        else:
            model = model.to(device)
        model = Inference.load_state_dict(args, model, pretrained_model_path)
        model.eval()
        
        if args.use_fp8:
            convert_fp8_linear(model)
        
        # ============================= Build extra models ========================
        # VAE
        print("="*25, f"load vae", "="*25)
        vae, _, s_ratio, t_ratio = load_vae(args.vae, 
                                            args.vae_precision, 
                                            logger=logger, 
                                            device='cpu' if args.cpu_offload else device,
                                            model_base=model_base)
        vae_kwargs = {'s_ratio': s_ratio, 't_ratio': t_ratio}
        
        # Parallel VAE
        device_vaes = []
        device_vaes.append(vae)
        if nccl_info.sp_size > 1 and nccl_info.rank_within_group == 0:
            for i in range(1, nccl_info.sp_size):
                cur_device = torch.device(f"cuda:{i}")
                # print("!!!!!!!!!! Load vae for ", cur_device)
                device_vae, _, _, _ = load_vae(args.vae, 
                                                args.vae_precision, 
                                                logger=logger, 
                                                device='cpu' if args.cpu_offload else cur_device,
                                                model_base=model_base)
                device_vaes.append(device_vae)
            vae.device_vaes = device_vaes
        
        # Text encoder
        if args.prompt_template_video is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get("crop_start", 0)
        else:
            crop_start = 0
        max_length = args.text_len + crop_start

        # prompt_template_video
        prompt_template_video = PROMPT_TEMPLATE[args.prompt_template_video] \
                                if args.prompt_template_video is not None else None
        print("="*25, f"load llava", "="*25)
        text_encoder = TextEncoder(text_encoder_type = args.text_encoder,
                                   max_length = max_length,
                                   text_encoder_precision = args.text_encoder_precision,
                                   tokenizer_type = args.tokenizer,
                                   use_attention_mask = args.use_attention_mask,
                                   prompt_template_video = prompt_template_video,
                                   hidden_state_skip_layer = args.hidden_state_skip_layer,
                                   apply_final_norm = args.apply_final_norm,
                                   reproduce = args.reproduce,
                                   logger = logger,
                                   device = 'cpu' if args.cpu_offload else device ,
                                   model_base=model_base
                                   )
        text_encoder_2 = None
        if args.text_encoder_2 is not None:
            text_encoder_2 = TextEncoder(text_encoder_type=args.text_encoder_2,
                                         max_length=args.text_len_2,
                                         text_encoder_precision=args.text_encoder_precision_2,
                                         tokenizer_type=args.tokenizer_2,
                                         use_attention_mask=args.use_attention_mask,
                                         reproduce=args.reproduce,
                                         logger=logger,
                                         device='cpu' if args.cpu_offload else device , 
                                         # if not args.use_cpu_offload else 'cpu'
                                         model_base=model_base
                                         )

        return cls(args=args, 
                   vae=vae, 
                   vae_kwargs=vae_kwargs, 
                   text_encoder=text_encoder,
                   model=model, 
                   text_encoder_2=text_encoder_2, 
                   device=device, 
                   logger=logger)

    @staticmethod
    def load_state_dict(args, model, ckpt_path):
        load_key = args.load_key
        ckpt_path = Path(ckpt_path)
        if ckpt_path.is_dir():
            ckpt_path = next(ckpt_path.glob("*_model_states.pt"))
        state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        if load_key in state_dict:
            state_dict = state_dict[load_key]
        elif load_key == ".":
            pass
        else:
            raise KeyError(f"Key '{load_key}' not found in the checkpoint. Existed keys: {state_dict.keys()}")
        model.load_state_dict(state_dict, strict=False)
        return model

    def get_exp_dir_and_ckpt_id(self):
        if self.ckpt is None:
            raise ValueError("The checkpoint path is not provided.")

        ckpt = Path(self.ckpt)
        if ckpt.parents[1].name == "checkpoints":
            # It should be a standard checkpoint path. We use the parent directory as the default save directory.
            exp_dir = ckpt.parents[2]
        else:
            raise ValueError(f"We cannot infer the experiment directory from the checkpoint path: {ckpt}. "
                             f"It seems that the checkpoint path is not standard. Please explicitly provide the "
                             f"save path by --save-path.")
        return exp_dir, ckpt.parent.name

    @staticmethod
    def parse_size(size):
        if isinstance(size, int):
            size = [size]
        if not isinstance(size, (list, tuple)):
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        if len(size) == 1:
            size = [size[0], size[0]]
        if len(size) != 2:
            raise ValueError(f"Size must be an integer or (height, width), got {size}.")
        return size
