import torch
import os
import numpy as np
import imageio
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union

from ...operators.astra_operator import AstraOperator
from ...synthesis.visual_generation.kling.astra_synthesis import AstraSynthesis
from ...memories.visual_synthesis.kling.astra_memory import AstraMemory


@dataclass
class AstraConfig:
    # pretrained model paths
    astra_path: str = ""
    wan_model_path: str = ""
    # default generation parameters
    start_frame: int = 0
    initial_condition_frames: int = 1
    frames_per_generation: int = 8
    total_frames_to_generate: int = 16  # frames per direction step; actual total = this * num_directions
    max_history_frames: int = 49        # sliding window size for memory
    # guidance parameters
    use_camera_cfg: bool = True
    camera_guidance_scale: float = 2.0
    text_guidance_scale: float = 1.0
    # MoE models parameters
    moe_num_experts: int = 3
    moe_top_k: int = 1
    moe_hidden_dim: Optional[int] = None
    # data parameters
    modality_type: str = "sekai"
    use_real_poses: bool = False
    scene_info_path: Optional[str] = None
    use_gt_prompt: bool = False
    # inference parameters
    device: str = "cuda"
    add_icons: bool = True


class AstraPipeline(object):
    def __init__(self, operator, synthesis, memory, config):
        self.operator = operator
        self.synthesis = synthesis
        self.memory = memory
        self.config = config
        self.device = synthesis.device

    @classmethod
    def from_pretrained(cls,
                        model_path: str,
                        required_components: dict,
                        device: str = "cuda",
                        **kwargs):
        """
        Load pretrained models and initialize the pipeline.
        """
        if isinstance(required_components, dict) and "wan_model_path" in required_components.keys():
            wan_model_path = required_components.get("wan_model_path", "Wan-AI/Wan2.1-T2V-1.3B")
        else:
            wan_model_path = "Wan-AI/Wan2.1-T2V-1.3B"
        config = AstraConfig(
            astra_path=model_path,
            wan_model_path=wan_model_path,
            device=device
        )
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

        print(f"Loading Astra components on {device}...")
        synthesis = AstraSynthesis.from_pretrained(config, device=device)
        operator = AstraOperator(device=device)
        memory = AstraMemory(capacity=config.max_history_frames)

        return cls(operator, synthesis, memory, config)

    def _stitch_camera_embeddings(
        self,
        direction_embeddings: List[torch.Tensor],
        initial_condition_frames: int,
        per_direction_frames: int,
        total_frames: int
    ) -> torch.Tensor:
        """
        Stitch per-direction camera embeddings into a single continuous embedding
        that supports absolute-position indexing used by the memory module.

        The memory's select() method reads camera_embedding_full[T : T + n] where
        T is the current history length. After the first direction's frames are
        generated, T advances past them and the next direction's poses must be
        at the correct absolute positions. This method achieves that by extracting
        each direction's motion portion and placing them sequentially.

        Layout of the combined embedding (example with 2 directions):
            [0, cond)                            : condition frame(s) from direction 0
            [cond, cond + per_dir)               : direction 0 motion poses
            [cond + per_dir, cond + 2 * per_dir) : direction 1 motion poses
            [cond + 2 * per_dir, ...)            : zero-padded (identity pose)

        Note: the mask column (last dim) is included as-is, but the memory module
        overrides it in prepare_framepack, so its values here do not matter.

        Args:
            direction_embeddings: Camera embedding per direction, each [T_i, emb_dim].
            initial_condition_frames: Number of condition frames at the beginning.
            per_direction_frames: Number of new motion frames per direction step.
            total_frames: Total motion frames across all directions.

        Returns:
            Combined camera embedding of shape [needed_length, emb_dim].
        """
        emb_dim = direction_embeddings[0].shape[1]
        dtype = direction_embeddings[0].dtype
        device = direction_embeddings[0].device

        # Allocate with sufficient padding for FramePack indexing
        needed_length = initial_condition_frames + total_frames + 30
        combined = torch.zeros(needed_length, emb_dim, dtype=dtype, device=device)

        # Copy condition frame(s) from the first direction's embedding
        cond_len = min(initial_condition_frames, direction_embeddings[0].shape[0])
        combined[:cond_len, :] = direction_embeddings[0][:cond_len, :]

        # Copy motion frames from each direction sequentially
        dst_offset = initial_condition_frames
        for emb in direction_embeddings:
            # In each individual embedding, motion occupies frames
            # [initial_condition_frames, initial_condition_frames + per_direction_frames)
            src_start = initial_condition_frames
            src_end = min(initial_condition_frames + per_direction_frames, emb.shape[0])
            n_motion = src_end - src_start
            combined[dst_offset:dst_offset + n_motion, :] = emb[src_start:src_end, :]
            dst_offset += n_motion

        return combined

    def process(self, input_: str, interaction: Dict[str, Any]):
        """
        Process the input image and interaction signals to prepare all data
        needed for autoregressive generation.

        Args:
            input_: Path to the condition image.
            interaction: Dict with keys:
                - 'prompt' (str): Text description of the scene.
                - 'direction' (str | List[str]): Camera direction(s).

        Returns:
            Dict containing initial_latents, prompt embeddings, stitched camera
            embeddings, and the computed total_frames.
        """
        args = self.config
        condition_image = input_
        prompt = interaction.get("prompt", "")
        directions = interaction.get("direction", "forward")

        # Normalize direction to a list for unified handling (backward compatible)
        if isinstance(directions, str):
            directions = [directions]

        num_directions = len(directions)
        per_direction_frames = args.total_frames_to_generate
        total_frames = per_direction_frames * num_directions

        # 1. Load and encode the condition image (Perception -> Representation)
        print(f"Processing image: {condition_image}")
        frames = self.operator.process_perception(condition_image=condition_image)
        latents = self.synthesis.encode_frames(frames)

        target_height, target_width = 60, 104
        C, T, H, W = latents.shape
        if H > target_height or W > target_width:
            h_start = (H - target_height) // 2
            w_start = (W - target_width) // 2
            latents = latents[:, :, h_start:h_start+target_height, w_start:w_start+target_width]

        model_dtype = next(self.synthesis.pipe.dit.parameters()).dtype
        history_latents = latents[:, :args.initial_condition_frames, :, :].to(self.device, dtype=model_dtype)
        initial_latents = history_latents  # backup for final concatenation

        # 2. Encode the text prompt (Interaction -> Representation)
        print(f"Encoding prompt: {prompt}")
        prompt_emb_pos = self.synthesis.pipe.encode_prompt(prompt)
        prompt_emb_neg = None
        if args.text_guidance_scale > 1.0:
            prompt_emb_neg = self.synthesis.pipe.encode_prompt("")

        # 3. Generate camera embeddings for each direction separately, then stitch.
        #
        #    Why separate calls: operator.process_interaction() only reads the LAST
        #    entry of self.current_interaction as the direction. So we must call it
        #    once per direction to obtain each direction's camera embedding.
        #
        #    Why stitch: memory.select() indexes into camera_embedding_full using the
        #    absolute history length T. After direction 0's frames are generated, T
        #    has advanced, so direction 1's poses must sit at the correct offset.
        print(f"Generating camera embeddings for directions: {directions} "
              f"({num_directions} step(s) x {per_direction_frames} frames "
              f"= {total_frames} total frames)...")

        direction_embeddings = []
        for direction in directions:
            # Set operator to target a single direction for this call
            self.operator.current_interaction = [direction]
            cam_emb = self.operator.process_interaction(
                modality_type=args.modality_type,
                start_frame=args.start_frame,
                initial_condition_frames=args.initial_condition_frames,
                total_frames_to_generate=per_direction_frames,
                use_real_poses=args.use_real_poses,
                scene_info_path=args.scene_info_path
            )
            direction_embeddings.append(cam_emb)

        # Stitch into one continuous embedding aligned with absolute frame positions
        camera_embedding_full = self._stitch_camera_embeddings(
            direction_embeddings,
            args.initial_condition_frames,
            per_direction_frames,
            total_frames
        )

        camera_embedding_uncond = None
        if args.use_camera_cfg:
            camera_embedding_uncond = torch.zeros_like(camera_embedding_full)

        return {
            "initial_latents": initial_latents,
            "prompt_emb_pos": prompt_emb_pos,
            "prompt_emb_neg": prompt_emb_neg,
            "camera_embedding_full": camera_embedding_full,
            "camera_embedding_uncond": camera_embedding_uncond,
            "total_frames": total_frames,
        }

    def __call__(self,
                 image_path: str,
                 interactions: Dict[str, Any],
                 images: Optional[List[Image.Image]] = None
                ) -> List[Image.Image]:
        """
        Main entry point. Supports multi-step direction interaction.

        When 'direction' is a list of N directions, the pipeline generates
        total_frames_to_generate * N frames, with camera motion transitioning
        through each direction sequentially. frames_per_generation (the
        autoregressive step size) remains unchanged.

        Args:
            image_path: Path to the condition image.
            interactions: Dict with keys:
                - 'prompt' (str): Text description of the scene.
                - 'direction' (str | List[str]): One or more camera directions.
                  Supported: forward, backward, left, right, forward_left,
                  forward_right, s_curve, left_right.

        Returns:
            List of PIL.Image frames comprising the generated video.
        """
        args = self.config

        # Process input and interaction signals
        processed_data = self.process(image_path, interactions)

        initial_latents = processed_data["initial_latents"]
        camera_embedding_full = processed_data["camera_embedding_full"]
        prompt_emb_pos = processed_data["prompt_emb_pos"]
        prompt_emb_neg = processed_data["prompt_emb_neg"]
        camera_embedding_uncond = processed_data["camera_embedding_uncond"]
        total_frames = processed_data["total_frames"]

        # Initialize memory with the condition frame
        self.memory.history_latents = None
        self.memory.record(initial_latents)

        total_generated = 0
        all_generated_frames = []

        # Autoregressive generation loop over all direction steps
        while total_generated < total_frames:
            current_generation = min(args.frames_per_generation, total_frames - total_generated)
            print(f"Generation step: {total_generated}/{total_frames}")

            framepack_data = self.memory.select(
                current_generation,
                camera_embedding_full,
                args.start_frame,
                args.modality_type
            )

            new_latents = self.synthesis.predict(
                framepack_data,
                current_generation,
                prompt_emb_pos,
                prompt_emb_neg,
                args,
                camera_embedding_uncond
            )

            new_latents_squeezed = new_latents.squeeze(0)
            self.memory.record(new_latents_squeezed)

            all_generated_frames.append(new_latents_squeezed)
            total_generated += current_generation

        # Concatenate all generated frames and prepend the initial condition frame
        all_generated = torch.cat(all_generated_frames, dim=1)
        final_video_latents = torch.cat(
            [initial_latents.to(all_generated.device), all_generated], dim=1
        ).unsqueeze(0)

        print("Decoding video...")
        video_np = self.synthesis.decode_video(final_video_latents)
        pil_frames = [Image.fromarray(frame) for frame in video_np]
        return pil_frames
