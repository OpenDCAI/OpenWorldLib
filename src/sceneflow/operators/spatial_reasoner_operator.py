from .base_operator import BaseOperator
from qwen_vl_utils import process_vision_info

class SpatialReasonerOperator(BaseOperator):
    """
    Lightweight operator placeholder for SpatialReasoner.
    Tracks interactions and encapsulates vision/text preprocessing for the pipeline.
    """

    def __init__(self, operation_types=None, interaction_template=None):
        super().__init__(operation_types=operation_types or ["reasoning"])
        self.interaction_template = interaction_template or []
        self.interaction_template_init()

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "SpatialReasonerOperator":
        return cls()

    def check_interaction(self, interaction):
        # Accept any interaction; extend validation if needed.
        return True

    def get_interaction(self, interaction):
        if self.check_interaction(interaction):
            self.current_interaction.append(interaction)


    def process_interaction(self, *args, **kwargs):
        return self.current_interaction

    def process_perception(self, batched_messages, texts, processor):
        """Process vision info and pack model inputs via processor."""
        vision_info = [process_vision_info(m) for m in batched_messages]
        image_inputs, video_inputs = [], []
        for imgs, vids in vision_info:
            image_inputs.append(imgs if imgs else None)
            video_inputs.append(vids if vids else None)
        if all(v is None for v in image_inputs):
            image_inputs = None
        if all(v is None for v in video_inputs):
            video_inputs = None

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def delete_last_interaction(self):
        super().delete_last_interaction()
