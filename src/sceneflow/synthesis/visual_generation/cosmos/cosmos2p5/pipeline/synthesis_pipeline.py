import math
from typing import Any, List, Optional, Union

import torch
from PIL import Image


class Cosmos2p5SynthesisPipeline:
    """High-level synthesis pipeline for Cosmos Predict 2.5.

    Responsibilities:
    - Run img2video generation in autoregressive chunks (default chunk size: 93 frames).
    - Use an `operator` for optional preprocessing of inputs.
    - Record each generated chunk into `memory` after generation.

    Notes:
    - This class is intentionally lightweight and expects an instance of the
      model (compatible with `cosmos2p5_predict_synthesis.Cosmos2p5PredictSynthesis`)
      to be passed in as `synthesis_model`.
    - `operator` is optional and should implement a preprocessing interface
      (e.g. `process_perception` or `preprocess`). If present, the pipeline
      will call it before the first chunk to obtain a conditioning image.
    - `memory` is optional and should implement a `record(item)` method.
    """

    def __init__(
        self,
        synthesis_model: Any,
        operator: Optional[Any] = None,
        memory: Optional[Any] = None,
        device: str = "cuda",
    ):
        self.synthesis_model = synthesis_model
        self.operator = operator
        self.memory = memory
        self.device = device

    @classmethod
    def from_components(cls, synthesis_model: Any, operator: Optional[Any] = None, memory: Optional[Any] = None, device: str = "cuda"):
        return cls(synthesis_model=synthesis_model, operator=operator, memory=memory, device=device)

    def _ensure_pil_list(self, video) -> List[Image.Image]:
        # Accept common return types and normalize to a list of PIL images.
        if video is None:
            return []
        # torch Tensor: assume shape [T, C, H, W] or [B, T, C, H, W]
        if isinstance(video, torch.Tensor):
            v = video.detach().cpu()
            if v.ndim == 5:  # [B, T, C, H, W]
                v = v[0]
            # now v is [T, C, H, W]
            frames = []
            for t in range(v.shape[0]):
                arr = (v[t].permute(1, 2, 0).numpy() * 255.0).astype('uint8')
                frames.append(Image.fromarray(arr))
            return frames
        # list/tuple of PIL images
        if isinstance(video, (list, tuple)):
            return list(video)
        # single PIL image
        if isinstance(video, Image.Image):
            return [video]
        # fallback: try to iterate
        try:
            return list(video)
        except Exception:
            raise TypeError("Unsupported video type returned from synthesis model")

    def generate_img2video_autoregressive(
        self,
        prompt: str,
        initial_image: Optional[Image.Image],
        total_frames: int,
        chunk_size: int = 93,
        height: int = 704,
        width: int = 1280,
        fps: int = 28,
        pad_mode: str = 'repeat',
        **synthesis_kwargs,
    ) -> List[Image.Image]:
        """Generate an image-conditioned video by autoregressively producing
        chunks of `chunk_size` frames until `total_frames` is reached.

        Behavior:
        - The first chunk is conditioned on `initial_image` (after operator preprocessing if available).
        - Each subsequent chunk is conditioned on the last frame of the previously-generated chunk.
        - After each chunk is generated, it is recorded via `memory.record(chunk)` if `memory` is provided.

        Returns a flat list of PIL frames (length == total_frames).
        """
        if total_frames <= 0:
            return []

        remaining = total_frames
        produced_frames: List[Image.Image] = []

        # Preprocess initial image via operator if available
        conditioning_image = initial_image
        if self.operator is not None:
            # prefer process_perception (common in repo), fall back to preprocess
            if hasattr(self.operator, 'process_perception'):
                try:
                    # expect a dict with an 'image' key or similar; accept PIL image fallback
                    proc = self.operator.process_perception(conditioning_image, num_output_frames=min(chunk_size, remaining), resize_H=height, resize_W=width, device=self.device)
                    if isinstance(proc, dict):
                        if 'image' in proc:
                            conditioning_image = proc['image']
                    else:
                        conditioning_image = proc
                except Exception:
                    # ignore operator error and continue with raw image
                    conditioning_image = initial_image
            elif hasattr(self.operator, 'preprocess'):
                try:
                    conditioning_image = self.operator.preprocess(conditioning_image)
                except Exception:
                    conditioning_image = initial_image

        while remaining > 0:
            cur_chunk = min(chunk_size, remaining)
            # Call the synthesis model. We pass `image` as conditioning and specify num_frames=cur_chunk.
            video_chunk = self.synthesis_model(
                prompt=prompt,
                image=conditioning_image,
                num_frames=cur_chunk,
                height=height,
                width=width,
                fps=fps,
                pad_mode=pad_mode,
                **synthesis_kwargs,
            )

            frames = self._ensure_pil_list(video_chunk)
            if len(frames) == 0:
                break

            # If model returned more frames than requested, trim to requested size
            if len(frames) > cur_chunk:
                frames = frames[:cur_chunk]

            # Record to memory if available
            if self.memory is not None and hasattr(self.memory, 'record'):
                try:
                    self.memory.record(frames)
                except Exception:
                    # ignore memory errors
                    pass

            # Append produced frames and prepare conditioning for next chunk
            produced_frames.extend(frames)
            # for autoregressive conditioning we use the last frame
            conditioning_image = frames[-1]

            remaining -= len(frames)

        # If we produced more frames than requested, trim
        if len(produced_frames) > total_frames:
            produced_frames = produced_frames[:total_frames]

        return produced_frames
