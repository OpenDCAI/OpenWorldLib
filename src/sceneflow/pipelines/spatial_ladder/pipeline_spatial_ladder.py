from typing import List, Optional, Sequence, Union

from qwen_vl_utils import process_vision_info

from ...reasoning.spatial_reasoning.spatial_ladder.spatial_ladder_reasoning import (
    SpatialLadderReasoning,
)


ImageLike = Union[str, bytes]
VideoLike = Union[str, bytes]


class SpatialLadderPipeline:
    """
    Minimal pipeline that wraps SpatialLadderReasoning for single-call inference.
    """

    def __init__(self, reasoning: SpatialLadderReasoning):
        self.reasoning = reasoning

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str = "hongxingli/SpatialLadder-3B",
        **kwargs,
    ) -> "SpatialLadderPipeline":
        reasoning = SpatialLadderReasoning.from_pretrained(
            pretrained_model_path=pretrained_model_path,
            **kwargs,
        )
        return cls(reasoning=reasoning)

    def _build_messages(
        self,
        image_paths: Optional[Union[ImageLike, Sequence[ImageLike]]],
        video_paths: Optional[Union[VideoLike, Sequence[VideoLike]]],
        instruction: str,
    ):
        if image_paths is None:
            image_paths = []
        if video_paths is None:
            video_paths = []
        if isinstance(image_paths, (str, bytes)):
            image_paths = [image_paths]
        if isinstance(video_paths, (str, bytes)):
            video_paths = [video_paths]

        content = [{"type": "image", "image": path} for path in image_paths]
        content += [{"type": "video", "video": path} for path in video_paths]
        content.append({"type": "text", "text": instruction})
        return [{"role": "user", "content": content}]

    def __call__(
        self,
        instruction: str,
        image_paths: Optional[Union[ImageLike, Sequence[ImageLike]]] = None,
        video_paths: Optional[Union[VideoLike, Sequence[VideoLike]]] = None,
        max_new_tokens: int = 2048,
        messages: Optional[list] = None,
        generation_kwargs: Optional[dict] = None,
    ) -> List[str]:
        if messages is None:
            batched_messages = [
                self._build_messages(
                    image_paths=image_paths,
                    video_paths=video_paths,
                    instruction=instruction,
                )
            ]
        else:
            if not messages:
                raise ValueError("messages must be non-empty.")
            batched_messages = [messages] if isinstance(messages[0], dict) else messages

        texts = [
            self.reasoning.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in batched_messages
        ]

        vision_info = [process_vision_info(m) for m in batched_messages]
        image_inputs, video_inputs = [], []
        for imgs, vids in vision_info:
            image_inputs.append(imgs if imgs else None)
            video_inputs.append(vids if vids else None)
        if all(v is None for v in image_inputs):
            image_inputs = None
        if all(v is None for v in video_inputs):
            video_inputs = None

        inputs = self.reasoning.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return self.reasoning.inference(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            generation_kwargs=generation_kwargs,
        )
