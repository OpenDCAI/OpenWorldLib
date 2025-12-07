import torch
from typing import Optional
from torchvision.transforms import v2
from ...operators.hunyuan_game_craft_operator import HunyuanGameCraftOperator
from ...synthesis.visual_generation.hunyuan_world.hunyuan_game_craft_synthesis import HunyuanGameCraftSynthesis
from ...synthesis.visual_generation.hunyuan_world.hunyuan_game_craft.config import parse_args
from ...synthesis.visual_generation.hunyuan_world.hunyuan_game_craft.modules.parallel_states import initialize_distributed

class HunyuanGameCraftPipeline:
    def __init__(self,
                operators: Optional[HunyuanGameCraftOperator] = None,
                synthesis_model: Optional[HunyuanGameCraftSynthesis] = None,
                device: str = "cuda",
                weight_dtype = torch.bfloat16
                ):
        self.synthesis_model = synthesis_model 
        self.operators = operators
        self.device = device
        self.weight_dtype = weight_dtype

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @classmethod
    def from_pretrained(cls,
                        synthesis_model_path: Optional[str] = None,
                        weight_dtype = torch.bfloat16,
                        device: str = "cuda",
                        cpu_offload: bool = False,
                        seed: int = 250160,
                        **kwargs) -> "HunyuanGameCraftPipeline":
        

        args = parse_args()
        args.cpu_offload = cpu_offload
        args.seed = seed

        initialize_distributed(args.seed)

        if synthesis_model_path is None:
            synthesis_model_path = "tencent/Hunyuan-GameCraft-1.0"

        print(f"Loading HunyuanGameCraft synthesis model from {synthesis_model_path}...")
        
        synthesis_model = HunyuanGameCraftSynthesis.from_pretrained(
            pretrained_model_path=synthesis_model_path,
            device=device if not args.cpu_offload else torch.device("cpu"),
            weight_dtype=weight_dtype,
            args=args,
            **kwargs
        )
        operators = HunyuanGameCraftOperator()

        pipeline = cls(
            operators=operators,
            synthesis_model=synthesis_model,
            device=device,
            weight_dtype=weight_dtype
        )
        return pipeline

    def _resizecrop(self, image, th, tw):
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        image = image.crop((left, top, right, bottom))
        return image
    
    def process(self,
                input_image,
                output_H,
                output_W,
                interaction_signal):
        """
        the input_image is PIL image
        """

        visual_context = self.operators.process_perception(image=input_image, output_H=output_H, output_W=output_W, process_model=self.synthesis_model)
        
        # define the interaction
        self.operators.get_interaction(interaction_signal)
        operator_condition = self.operators.process_interaction()
        self.operators.delete_last_interaction()

        output_dict = {
            "visual_context": visual_context,
            "operator_condition": operator_condition,
        }
        
        return output_dict

    def __call__(self,
                # condition
                input_image,
                interaction_signal=["forward", "left", "right", "right", "camera_l", "camera_r", "camera_up", "camera_down"],
                interaction_speed=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                interaction_text_prompt="",
                interaction_positive_prompt="Realistic, High-quality.",
                interaction_negative_prompt="overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.",
                # generation config
                output_H=704,
                output_W=1216,
                num_output_frames=129,
                cfg_scale=2.0,
                infer_steps=50,
                flow_shift_eval_video=5.0,
                **kwds):

        output_dict = self.process(
            input_image=input_image,
            output_H=output_H,
            output_W=output_W,
            interaction_signal=interaction_signal
        )
        output_video = self.synthesis_model.predict(
            # condition
            ref_images=output_dict["visual_context"]['ref_images'],
            last_latents=output_dict["visual_context"]['last_latents'],
            ref_latents=output_dict["visual_context"]['ref_latents'],
            action_list=output_dict['operator_condition'],
            action_speed_list=interaction_speed,
            prompt=interaction_text_prompt,
            negative_prompt=interaction_negative_prompt,
            # generation config
            size=(output_H, output_W),
            video_length=num_output_frames,
            guidance_scale=cfg_scale,
            infer_steps=infer_steps,
            flow_shift=flow_shift_eval_video,
            **kwds
        )
        return output_video
