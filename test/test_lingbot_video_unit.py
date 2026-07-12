import unittest

import numpy as np
from PIL import Image

from openworldlib.memories.visual_synthesis.lingbot_video import LingBotVideoMemory
from openworldlib.operators.lingbot_video_operator import LingBotVideoOperator
from openworldlib.pipelines.lingbot_video import LingBotVideoPipeline
from openworldlib.reasoning.visual_reasoning.lingbot_video import LingBotVideoReasoning


class _Output:
    def __init__(self, frames):
        self.frames = frames


class _Synthesis:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        frame_count = 1 if kwargs["processed_inputs"]["mode"] == "t2i" else 5
        return _Output([np.zeros((frame_count, 32, 48, 3), dtype=np.float32)])


class _RewriteBackend:
    def __init__(self):
        self.calls = []

    def generate(self, text, image, use_lora):
        self.calls.append((text, image, use_lora))
        return '{"scene":"expanded"}' if use_lora else "expanded description"


def _pipeline(mode, reasoning_model=None):
    return LingBotVideoPipeline(
        operator=LingBotVideoOperator(),
        synthesis_model=_Synthesis(),
        memory_module=LingBotVideoMemory(),
        reasoning_model=reasoning_model,
        mode=mode,
    )


class LingBotVideoPipelineTest(unittest.TestCase):
    def test_t2i_call_records_a_reusable_pil_frame(self):
        pipeline = _pipeline("t2i")
        pipeline(prompt="still image")
        self.assertIsInstance(pipeline.memory_module.select(type="image"), Image.Image)

    def test_t2v_call_routes_video_mode(self):
        pipeline = _pipeline("t2v")
        pipeline(prompt="moving scene", num_frames=5)
        call = pipeline.synthesis_model.calls[-1]
        self.assertEqual(call["processed_inputs"]["mode"], "t2v")
        self.assertEqual(call["num_frames"], 5)

    def test_i2v_alias_routes_ti2v_with_existing_asset_type(self):
        pipeline = _pipeline("i2v")
        image = Image.new("RGB", (48, 32), "white")
        pipeline(prompt="animate", images=image, num_frames=5)
        processed = pipeline.synthesis_model.calls[-1]["processed_inputs"]
        self.assertEqual(processed["mode"], "ti2v")
        self.assertIsInstance(processed["image"], Image.Image)

    def test_mode_cannot_change_after_loading(self):
        pipeline = _pipeline("t2v")
        with self.assertRaisesRegex(ValueError, "loaded for mode"):
            pipeline(prompt="animate", mode="i2v", images=Image.new("RGB", (8, 8)))

    def test_i2v_stream_reuses_the_last_generated_frame(self):
        pipeline = _pipeline("i2v")
        pipeline.stream(prompt="first", images=Image.new("RGB", (48, 32)), num_frames=5)
        pipeline.stream(prompt="second", num_frames=5)
        second_input = pipeline.synthesis_model.calls[-1]["processed_inputs"]["image"]
        self.assertIsInstance(second_input, Image.Image)

    def test_prompt_rewriter_is_used_as_reasoning(self):
        backend = _RewriteBackend()
        pipeline = _pipeline("t2v", LingBotVideoReasoning(backend))
        pipeline(prompt="short prompt", rewrite_prompt=True)
        processed = pipeline.synthesis_model.calls[-1]["processed_inputs"]
        self.assertEqual(processed["prompt"], '{"scene":"expanded"}')
        self.assertEqual([call[2] for call in backend.calls], [False, True])


if __name__ == "__main__":
    unittest.main()
