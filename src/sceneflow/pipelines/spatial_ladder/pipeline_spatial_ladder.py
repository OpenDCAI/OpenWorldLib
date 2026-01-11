from typing import List, Optional, Sequence, Union

from ...reasoning.spatial_reasoning.spatial_ladder.spatial_ladder_reasoning import (
    SpatialLadderReasoning,
)
from ...operators.spatial_ladder_operator import SpatialLadderOperator


ImageLike = Union[str, bytes]
VideoLike = Union[str, bytes]


class SpatialLadderPipeline:
    """
    Pipeline that builds vision/text inputs and calls SpatialLadderReasoning directly.
    """

    def __init__(self, reasoning: SpatialLadderReasoning, operator: SpatialLadderOperator):
        self.reasoning = reasoning
        self.operator = operator
        self.processor = reasoning.processor

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
        operator = SpatialLadderOperator.from_pretrained()
        return cls(reasoning=reasoning, operator=operator)

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
        # Record interaction for interface consistency
        self.operator.get_interaction(instruction)
        self.operator.process_interaction()

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
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in batched_messages
        ]

        inputs = self.operator.process_perception(batched_messages, texts, processor=self.processor)

        outputs = self.reasoning.inference(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            generation_kwargs=generation_kwargs,
        )
        # Clean up current interaction entry
        self.operator.delete_last_interaction()
        return outputs
