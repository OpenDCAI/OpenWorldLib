# import pytorch_lightning as pl
import lightning as L
from lightning.pytorch.callbacks import Callback
import sys, gc
import random
import torch
import torchaudio
import typing as tp
import wandb
from aeiou.viz import audio_spectrogram_image
from ema_pytorch import EMA
from einops import rearrange
from safetensors.torch import save_file
from torch import optim
from torch.nn import functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from ..inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler
from ..models.diffusion import DiffusionModelWrapper, ConditionedDiffusionModelWrapper
from ..models.autoencoders import DiffusionAutoencoder
from .autoencoders import create_loss_modules_from_bottleneck
from .losses import MSELoss, MultiLoss
from .utils import create_optimizer_from_config, create_scheduler_from_config, generate_mask, generate_channel_mask
import os
from pathlib import Path
from time import time
import numpy as np

class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep

class DiffusionCondTrainingWrapper(L.LightningModule):
    '''
    Wrapper for training a conditional audio diffusion model.
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            lr: float = None,
            mask_padding: bool = False,
            mask_padding_dropout: float = 0.0,
            use_ema: bool = True,
            log_loss_info: bool = False,
            optimizer_configs: dict = None,
            diffusion_objective: tp.Literal["rectified_flow", "v"] = "v",
            pre_encoded: bool = False,
            cfg_dropout_prob = 0.1,
            timestep_sampler: tp.Literal["uniform", "logit_normal"] = "uniform",
            max_mask_segments = 0,
    ):
        super().__init__()

        self.diffusion = model

        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.diffusion_ema = None

        self.mask_padding = mask_padding
        self.mask_padding_dropout = mask_padding_dropout

        self.cfg_dropout_prob = cfg_dropout_prob

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler

        self.diffusion_objective = model.diffusion_objective
        print(f'Training in the {self.diffusion_objective} formulation with timestep sampler: {timestep_sampler}')

        self.max_mask_segments = max_mask_segments
            
        self.loss_modules = [
            MSELoss("output", 
                   "targets", 
                   weight=1.0, 
                   mask_key="padding_mask" if self.mask_padding else None, 
                   name="mse_loss"
            )
        ]

        self.losses = MultiLoss(self.loss_modules)

        self.log_loss_info = log_loss_info

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "diffusion": {
                    "optimizer": {
                        "type": "Adam",
                        "config": {
                            "lr": lr
                        }
                    }
                }
            }
        else:
            if lr is not None:
                print(f"WARNING: learning_rate and optimizer_configs both specified in config. Ignoring learning_rate and using optimizer_configs.")

        self.optimizer_configs = optimizer_configs

        self.pre_encoded = pre_encoded

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']
        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], self.diffusion.parameters())

        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            return [opt_diff], [sched_diff_config]

        return [opt_diff]

    def training_step(self, batch, batch_idx):
        reals, metadata = batch
        # import ipdb
        # ipdb.set_trace()
        p = Profiler()
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        diffusion_input = reals
        if not self.pre_encoded:
            loss_info["audio_reals"] = diffusion_input

        p.tick("setup")

        with torch.amp.autocast('cuda'):

            conditioning = self.diffusion.conditioner(metadata, self.device)
            

        video_exist = torch.stack([item['video_exist'] for item in metadata],dim=0)
        conditioning['metaclip_features'][~video_exist] = self.diffusion.model.model.empty_clip_feat
        conditioning['sync_features'][~video_exist] = self.diffusion.model.model.empty_sync_feat
        # If mask_padding is on, randomly drop the padding masks to allow for learning silence padding
        use_padding_mask = self.mask_padding and random.random() > self.mask_padding_dropout

        # Create batch tensor of attention masks from the "mask" field of the metadata array
        if use_padding_mask:
            padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device) # Shape (batch_size, sequence_length)

        p.tick("conditioning")

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.amp.autocast('cuda') and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)
                    
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
                    p.tick("pretransform")

                    # If mask_padding is on, interpolate the padding masks to the size of the pretransformed input
                    if use_padding_mask:
                        padding_masks = F.interpolate(padding_masks.unsqueeze(1).float(), size=diffusion_input.shape[2], mode="nearest").squeeze(1).bool()
            else:            
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        if self.max_mask_segments > 0:
            # Max mask size is the full sequence length
            max_mask_length = diffusion_input.shape[2]

            # Create a mask of random length for a random slice of the input
            masked_input, mask = self.random_mask(diffusion_input, max_mask_length)

            conditioning['inpaint_mask'] = [mask]
            conditioning['inpaint_masked_input'] = masked_input

        if self.timestep_sampler == "uniform":
            # Draw uniformly distributed continuous timesteps
            t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(reals.shape[0], device=self.device))
        # import ipdb
        # ipdb.set_trace()
        # Calculate the noise schedule parameters for those timesteps
        if self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)
        elif self.diffusion_objective == "rectified_flow":
            alphas, sigmas = 1-t, t

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas

        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif self.diffusion_objective == "rectified_flow":
            targets = noise - diffusion_input

        p.tick("noise")

        extra_args = {}

        if use_padding_mask:
            extra_args["mask"] = padding_masks

        with torch.amp.autocast('cuda'):
            p.tick("amp")
            output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = self.cfg_dropout_prob, **extra_args)
            p.tick("diffusion")

            loss_info.update({
                "output": output,
                "targets": targets,
                "padding_mask": padding_masks if use_padding_mask else None,
            })

            loss, losses = self.losses(loss_info)

            p.tick("loss")

            if self.log_loss_info:
                # Loss debugging logs
                num_loss_buckets = 10
                bucket_size = 1 / num_loss_buckets
                loss_all = F.mse_loss(output, targets, reduction="none")

                sigmas = rearrange(self.all_gather(sigmas), "w b c n -> (w b) c n").squeeze()

                # gather loss_all across all GPUs
                loss_all = rearrange(self.all_gather(loss_all), "w b c n -> (w b) c n")

                # Bucket loss values based on corresponding sigma values, bucketing sigma values by bucket_size
                loss_all = torch.stack([loss_all[(sigmas >= i) & (sigmas < i + bucket_size)].mean() for i in torch.arange(0, 1, bucket_size).to(self.device)])

                # Log bucketed losses with corresponding sigma bucket values, if it's not NaN
                debug_log_dict = {
                    f"model/loss_all_{i/num_loss_buckets:.1f}": loss_all[i].detach() for i in range(num_loss_buckets) if not torch.isnan(loss_all[i])
                }

                self.log_dict(debug_log_dict)


        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': diffusion_input.std(),
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        p.tick("log")
        #print(f"Profiler: {p}")
        return loss
    
    def validation_step(self, batch, batch_idx):
        reals, metadata = batch
        # breakpoint()
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        diffusion_input = reals

        if not self.pre_encoded:
            loss_info["audio_reals"] = diffusion_input


        with torch.amp.autocast('cuda'):

            conditioning = self.diffusion.conditioner(metadata, self.device)
        
        video_exist = torch.stack([item['video_exist'] for item in metadata],dim=0)
        conditioning['metaclip_features'][~video_exist] = self.diffusion.model.model.empty_clip_feat
        conditioning['sync_features'][~video_exist] = self.diffusion.model.model.empty_sync_feat

        if self.diffusion.pretransform is not None:

            if not self.pre_encoded:
                self.diffusion.pretransform.to(self.device)
                with torch.amp.autocast('cuda') and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)
                    
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
            else:            
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale
        if self.max_mask_segments > 0:
            # Max mask size is the full sequence length
            max_mask_length = diffusion_input.shape[2]

            # Create a mask of random length for a random slice of the input
            masked_input, mask = self.random_mask(diffusion_input, max_mask_length)

            conditioning['inpaint_mask'] = [mask]
            conditioning['inpaint_masked_input'] = masked_input
        if self.timestep_sampler == "uniform":
            # Draw uniformly distributed continuous timesteps
            t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(reals.shape[0], device=self.device))
            
        # Calculate the noise schedule parameters for those timesteps
        if self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)
        elif self.diffusion_objective == "rectified_flow":
            alphas, sigmas = 1-t, t

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)
        noised_inputs = diffusion_input * alphas + noise * sigmas

        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif self.diffusion_objective == "rectified_flow":
            targets = noise - diffusion_input


        with torch.amp.autocast('cuda'):
            output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = 0.0)

            loss_info.update({
                "output": output,
                "targets": targets,
            })

            loss, losses = self.losses(loss_info)


        log_dict = {
            'val_loss': loss.detach(),
        }

        self.log_dict(log_dict, prog_bar=True, batch_size=diffusion_input.size(0))

    def predict_step(self, batch, batch_idx):
        reals, metadata = batch
        ids = [item['id'] for item in metadata]
        batch_size, length = reals.shape[0], reals.shape[2]
        print(f"Predicting {batch_size} samples with length {length} for ids: {ids}")
        with torch.amp.autocast('cuda'):
            conditioning = self.diffusion.conditioner(metadata, self.device)
        
        video_exist = torch.stack([item['video_exist'] for item in metadata],dim=0)
        conditioning['metaclip_features'][~video_exist] = self.diffusion.model.model.empty_clip_feat
        conditioning['sync_features'][~video_exist] = self.diffusion.model.model.empty_sync_feat

        cond_inputs = self.diffusion.get_conditioning_inputs(conditioning)
        if batch_size > 1:
            noise_list = []
            for _ in range(batch_size):
                noise_1 = torch.randn([1, self.diffusion.io_channels, length]).to(self.device)  # 每次生成推进RNG状态
                noise_list.append(noise_1)
            noise = torch.cat(noise_list, dim=0)
        else:
            noise = torch.randn([batch_size, self.diffusion.io_channels, length]).to(self.device)
        with torch.amp.autocast('cuda'):

            model = self.diffusion.model
            if self.diffusion_objective == "v":
                fakes = sample(model, noise, 24, 0, **cond_inputs, cfg_scale=5, batch_cfg=True)
            elif self.diffusion_objective == "rectified_flow":
                import time
                start_time = time.time()
                fakes = sample_discrete_euler(model, noise, 24, **cond_inputs, cfg_scale=5, batch_cfg=True)
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"执行时间: {execution_time:.2f} 秒")
            if self.diffusion.pretransform is not None:
                fakes = self.diffusion.pretransform.decode(fakes)

        audios = fakes.to(torch.float32).div(torch.max(torch.abs(fakes))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        return audios
        # # Put the demos together
        # fakes = rearrange(fakes, 'b d n -> d (b n)')

    def random_mask(self, sequence, max_mask_length):
        b, _, sequence_length = sequence.size()

        # Create a mask tensor for each batch element
        masks = []

        for i in range(b):
            mask_type = random.randint(0, 2)

            if mask_type == 0:  # Random mask with multiple segments
                num_segments = random.randint(1, self.max_mask_segments)
                max_segment_length = max_mask_length // num_segments

                segment_lengths = random.sample(range(1, max_segment_length + 1), num_segments)
               
                mask = torch.ones((1, 1, sequence_length))
                for length in segment_lengths:
                    mask_start = random.randint(0, sequence_length - length)
                    mask[:, :, mask_start:mask_start + length] = 0

            elif mask_type == 1:  # Full mask
                mask = torch.zeros((1, 1, sequence_length))

            elif mask_type == 2:  # Causal mask
                mask = torch.ones((1, 1, sequence_length))
                mask_length = random.randint(1, max_mask_length)
                mask[:, :, -mask_length:] = 0

            mask = mask.to(sequence.device)
            masks.append(mask)

        # Concatenate the mask tensors into a single tensor
        mask = torch.cat(masks, dim=0).to(sequence.device)

        # Apply the mask to the sequence tensor for each batch element
        masked_sequence = sequence * mask

        return masked_sequence, mask

    def on_before_zero_grad(self, *args, **kwargs):
        if self.diffusion_ema is not None:
            self.diffusion_ema.update()

    def export_model(self, path, use_safetensors=False):
        if self.diffusion_ema is not None:
            self.diffusion.model = self.diffusion_ema.ema_model
        
        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)

class DiffusionCondDemoCallback(Callback):
    def __init__(self, 
                 demo_every=2000,
                 num_demos=8,
                 sample_size=65536,
                 demo_steps=250,
                 sample_rate=48000,
                 demo_conditioning: tp.Optional[tp.Dict[str, tp.Any]] = {},
                 demo_cfg_scales: tp.Optional[tp.List[int]] = [3, 5, 7],
                 demo_cond_from_batch: bool = False,
                 display_audio_cond: bool = False
    ):
        super().__init__()

        self.demo_every = demo_every
        self.num_demos = num_demos
        self.demo_samples = sample_size
        self.demo_steps = demo_steps
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.demo_conditioning = demo_conditioning
        self.demo_cfg_scales = demo_cfg_scales

        # If true, the callback will use the metadata from the batch to generate the demo conditioning
        self.demo_cond_from_batch = demo_cond_from_batch

        # If true, the callback will display the audio conditioning
        self.display_audio_cond = display_audio_cond

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: DiffusionCondTrainingWrapper, outputs, batch, batch_idx):        

        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        module.eval()
        
        print(f"Generating demo")
        self.last_demo_step = trainer.global_step

        demo_samples = self.demo_samples
        
        demo_cond = self.demo_conditioning

        if self.demo_cond_from_batch:
            # Get metadata from the batch
            demo_cond = batch[1][:self.num_demos]

        if '.pth' in demo_cond[0]:
            demo_cond_data = []
            for path in demo_cond:
                # info = {}
                data = torch.load(path, weights_only=True)
                if 'caption_t5' not in data.keys():
                    data['caption_t5'] = data['caption']
                data['seconds_start'] = 0
                data['seconds_total'] = 10
                demo_cond_data.append(data)
            demo_cond = demo_cond_data
        elif '.npz' in demo_cond[0]:
            demo_cond_data = []
            for path in demo_cond:
                # info = {}
                npz_data = np.load(path,allow_pickle=True)
                data = {key: npz_data[key] for key in npz_data.files}
                for key in data.keys():
                    # print(key)
                    if isinstance(data[key], np.ndarray) and np.issubdtype(data[key].dtype, np.number):
                        data[key] = torch.from_numpy(data[key])

                demo_cond_data.append(data)
            demo_cond = demo_cond_data
        if module.diffusion.pretransform is not None:
            demo_samples = demo_samples // module.diffusion.pretransform.downsampling_ratio

        noise = torch.randn([self.num_demos, module.diffusion.io_channels, demo_samples]).to(module.device)
            
        try:
            print("Getting conditioning")
            with torch.amp.autocast('cuda'):
                conditioning = module.diffusion.conditioner(demo_cond, module.device)

            cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)

            log_dict = {}

            if self.display_audio_cond:
                audio_inputs = torch.cat([cond["audio"] for cond in demo_cond], dim=0)
                audio_inputs = rearrange(audio_inputs, 'b d n -> d (b n)')

                filename = f'demo_audio_cond_{trainer.global_step:08}.wav'
                audio_inputs = audio_inputs.to(torch.float32).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, audio_inputs, self.sample_rate)
                log_dict[f'demo_audio_cond'] = wandb.Audio(filename, sample_rate=self.sample_rate, caption="Audio conditioning")
                log_dict[f"demo_audio_cond_melspec_left"] = wandb.Image(audio_spectrogram_image(audio_inputs))
                trainer.logger.experiment.log(log_dict)

            for cfg_scale in self.demo_cfg_scales:

                print(f"Generating demo for cfg scale {cfg_scale}")
                
                with torch.amp.autocast('cuda'):
                    # model = module.diffusion_ema.model if module.diffusion_ema is not None else module.diffusion.model
                    model = module.diffusion.model

                    if module.diffusion_objective == "v":
                        fakes = sample(model, noise, self.demo_steps, 0, **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True)
                    elif module.diffusion_objective == "rectified_flow":
                        fakes = sample_discrete_euler(model, noise, self.demo_steps, **cond_inputs, cfg_scale=cfg_scale, batch_cfg=True)
                    
                    if module.diffusion.pretransform is not None:
                        fakes = module.diffusion.pretransform.decode(fakes)

                # Put the demos together
                fakes = rearrange(fakes, 'b d n -> d (b n)')

                log_dict = {}
                
                filename = f'demos/demo_cfg_{cfg_scale}_{trainer.global_step:08}.wav'
                fakes = fakes.div(torch.max(torch.abs(fakes))).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, fakes, self.sample_rate)

                log_dict[f'demo_cfg_{cfg_scale}'] = wandb.Audio(filename,
                                                    sample_rate=self.sample_rate,
                                                    caption=f'Reconstructed')
            
                log_dict[f'demo_melspec_left_cfg_{cfg_scale}'] = wandb.Image(audio_spectrogram_image(fakes))
                trainer.logger.experiment.log(log_dict)
            
            del fakes

        except Exception as e:
            raise e
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            module.train()
