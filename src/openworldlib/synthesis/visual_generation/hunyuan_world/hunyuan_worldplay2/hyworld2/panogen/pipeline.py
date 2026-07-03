"""
HunyuanImage-3.0 Panorama Pipeline

Usage:
    # Python API — local path
    from pipeline import HunyuanPanoPipeline
    pipeline = HunyuanPanoPipeline.from_pretrained('./HunyuanImage-3')
    output = pipeline('input.png')
    output.save('output_panorama.png')

    # Python API — download from HuggingFace
    pipeline = HunyuanPanoPipeline.from_pretrained('tencent/HY-Pano-2.0')
    output = pipeline(
        'input.png',
        prompt='Expand this image to a 360-degree equirectangular panorama. Sunny day.',
        height=960, width=1952, seed=42)
    output.save('output_panorama.png')

    # CLI — Basic panorama generation
    python pipeline.py --image input.png

    # CLI — Specify prompt and output path
    python pipeline.py --image input.png \\
        --prompt "Expand this image to a 360-degree equirectangular panorama. Maintain realistic style." \\
        --save output_panorama.png

    # CLI — Customize inference steps, task type, and system prompt
    python pipeline.py --image input.png \\
        --diff-infer-steps 50 --bot-task think_recaption --use-system-prompt en_unified

    # CLI — Reproducible generation with a fixed seed
    python pipeline.py --image input.png --seed 42 --reproduce

    # CLI — Use Taylor Cache to speed up sampling
    python pipeline.py --image input.png \\
        --use-taylor-cache --taylor-cache-interval 5 --taylor-cache-order 2
"""

import argparse
import os
import numpy as np
from pathlib import Path
from PIL import Image
from hunyuan_image_3 import HunyuanImage3ForCausalMM

PANO_INSTRUCTION = "Expand this image to a 360-degree equirectangular panorama."


# ============================================================
# Utility functions
# ============================================================

def circular_blend_edges(image, blend_width=32):
    """Blend the left and right edges of an image for seamless panorama."""
    image = np.array(image)
    for x in range(blend_width):
        image[:, x, :] = (
            image[:, -blend_width + x, :] * (1 - x / blend_width) +
            image[:, x, :] * (x / blend_width)
        )
    return Image.fromarray(image[:, :-blend_width].astype(np.uint8))


def set_reproducibility(enable, global_seed=None, benchmark=None):
    import torch
    if enable:
        import random
        random.seed(global_seed)
        import numpy as np
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)
    if enable:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = (not enable) if benchmark is None else benchmark
    torch.backends.cudnn.deterministic = enable
    torch.use_deterministic_algorithms(enable)


def _has_model_files(path: str) -> bool:
    """Check whether a directory looks like a valid HunyuanImage-3 model dir."""
    return os.path.isdir(path) and any(
        os.path.isfile(os.path.join(path, name))
        for name in ("config.json", "tokenizer_config.json", "pytorch_model.bin")
    )


def _resolve_model_dir(pretrained_model_name_or_path: str, subfolder: str) -> str:
    """Resolve the local directory that contains model weights.

    Resolution order:
      1. {pretrained_model_name_or_path}/{subfolder}  — local repo root with subfolder
      2. {pretrained_model_name_or_path}              — direct local path
      3. HuggingFace download: snapshot_download(repo_id, allow_patterns=[subfolder/*])
    """
    candidate = os.path.join(pretrained_model_name_or_path, subfolder)
    if _has_model_files(candidate):
        print(f"[Init] Found local model at {candidate}")
        return candidate

    if _has_model_files(pretrained_model_name_or_path):
        print(f"[Init] Found local model at {pretrained_model_name_or_path}")
        return pretrained_model_name_or_path

    print(f"[Init] Downloading from HuggingFace: {pretrained_model_name_or_path} (subfolder={subfolder})")
    from huggingface_hub import snapshot_download
    repo_root = snapshot_download(
        repo_id=pretrained_model_name_or_path,
        allow_patterns=[f"{subfolder}/*"],
    )
    resolved = os.path.join(repo_root, subfolder)
    if not _has_model_files(resolved):
        raise FileNotFoundError(
            f"Downloaded repo '{pretrained_model_name_or_path}' but subfolder '{subfolder}' "
            f"does not contain expected model files. "
            f"Check that the repo and subfolder name are correct."
        )
    return resolved


# ============================================================
# Pipeline
# ============================================================

class HunyuanPanoPipeline:
    """
    HunyuanImage-3.0 Panorama generation pipeline.

    Model-level parameters (fixed after loading) are passed to __init__ /
    from_pretrained.  Per-inference parameters are passed to forward / __call__.
    """

    DEFAULT_MODEL_ID = "tencent/HY-World-2.0"
    DEFAULT_SUBFOLDER = "HY-Pano-2.0"

    def __init__(self, model, *, attn_impl="sdpa", moe_impl="eager"):
        """
        Args:
            model:     Pre-loaded HunyuanImage3ForCausalMM instance.
            attn_impl: Attention implementation used when the model was loaded.
            moe_impl:  MoE implementation used when the model was loaded.
        """
        self.model = model
        self.attn_impl = attn_impl
        self.moe_impl = moe_impl

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_ID,
        *,
        subfolder: str = DEFAULT_SUBFOLDER,
        attn_impl: str = "sdpa",
        moe_impl: str = "eager",
    ) -> "HunyuanPanoPipeline":
        """Load model weights and return a ready-to-use pipeline.

        Args:
            pretrained_model_name_or_path: HuggingFace repo ID or local path.
                Model files are expected under ``{path}/{subfolder}/`` or
                directly under ``{path}/`` (for backward compatibility).
            subfolder: Subfolder inside the repo that contains the model
                checkpoint. Defaults to 'HY-Pano-2.0'.
            attn_impl: Attention implementation ('sdpa' or 'flash_attention_2').
            moe_impl:  MoE implementation ('eager' or 'flashinfer').
        """
        model_dir = _resolve_model_dir(pretrained_model_name_or_path, subfolder)

        print(f"[Init] Loading model from {model_dir} ...")
        kwargs = dict(
            attn_implementation=attn_impl,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            moe_impl=moe_impl,
            moe_drop_tokens=True,
        )
        model = HunyuanImage3ForCausalMM.from_pretrained(model_dir, **kwargs)
        model.load_tokenizer(model_dir)
        print("[Init] Model loaded successfully!")

        return cls(model, attn_impl=attn_impl, moe_impl=moe_impl)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(
        self,
        image,
        *,
        prompt=PANO_INSTRUCTION,
        seed=None,
        height=960,
        width=1952,
        use_system_prompt="en_unified",
        system_prompt=None,
        bot_task="think_recaption",
        diff_infer_steps=50,
        verbose=2,
        max_new_tokens=2048,
        infer_align_image_size=False,
        blend_width=32,
        # Taylor Cache
        use_taylor_cache=False,
        taylor_cache_interval=5,
        taylor_cache_order=2,
        taylor_cache_enable_first_enhance=False,
        taylor_cache_first_enhance_steps=3,
        taylor_cache_enable_tailing_enhance=False,
        taylor_cache_tailing_enhance_steps=1,
        taylor_cache_low_freqs_order=2,
        taylor_cache_high_freqs_order=2,
    ):
        """Run panorama generation and return the blended output image.

        Args:
            image:  Path to the input image (str or Path).
            prompt: Text prompt. The panorama instruction is prepended
                    automatically if not already present.
            seed:   Random seed (None = random).
            height: Output image height in pixels.
            width:  Output image width in pixels.
            use_system_prompt: System prompt type.
            system_prompt: Custom system prompt text (used when
                           use_system_prompt='custom').
            bot_task: Task type ('image', 'auto', 'recaption',
                      'think_recaption').
            diff_infer_steps: Number of diffusion inference steps.
            verbose: Verbosity level (0 / 1 / 2).
            max_new_tokens: Maximum number of new tokens to generate.
            infer_align_image_size: Align output size to input image size.
            blend_width: Edge blending width for seamless panorama.
            use_taylor_cache: Enable Taylor Cache for faster sampling.
            taylor_cache_interval: Taylor Cache update interval.
            taylor_cache_order: Taylor Cache polynomial order.
            taylor_cache_enable_first_enhance: Enable first-step enhancement.
            taylor_cache_first_enhance_steps: Steps for first-step enhancement (>2).
            taylor_cache_enable_tailing_enhance: Enable tailing enhancement.
            taylor_cache_tailing_enhance_steps: Steps for tailing enhancement.
            taylor_cache_low_freqs_order: Low-frequency order in Taylor Cache.
            taylor_cache_high_freqs_order: High-frequency order in Taylor Cache.

        Returns:
            PIL.Image: The generated panorama image (after edge blending).
        """
        image = str(image)
        if not Path(image).exists():
            raise ValueError(f"Input image does not exist: {image}")

        # Ensure the panorama instruction is always present
        if PANO_INSTRUCTION not in prompt:
            prompt = f"{PANO_INSTRUCTION} {prompt}".strip()
            print(f"[Info] Panorama instruction prepended. Final prompt: {prompt}")

        print("Start generating panorama image...")
        print(f"  Input image:     {image}")
        print(f"  Prompt:          {prompt}")
        print(f"  Infer steps:     {diff_infer_steps}")
        print(f"  Seed:            {seed}")
        print(f"  Task type:       {bot_task}")
        print(f"  System prompt:   {use_system_prompt}")

        cot_text, samples = self.model.generate_image(
            prompt=prompt,
            image=[image],
            seed=seed,
            image_size=[height, width],
            use_system_prompt=use_system_prompt,
            system_prompt=system_prompt,
            bot_task=bot_task,
            diff_infer_steps=diff_infer_steps,
            verbose=verbose,
            max_new_tokens=max_new_tokens,
            infer_align_image_size=infer_align_image_size,
            use_taylor_cache=use_taylor_cache,
            taylor_cache_interval=taylor_cache_interval,
            taylor_cache_order=taylor_cache_order,
            taylor_cache_enable_first_enhance=taylor_cache_enable_first_enhance,
            taylor_cache_first_enhance_steps=taylor_cache_first_enhance_steps,
            taylor_cache_enable_tailing_enhance=taylor_cache_enable_tailing_enhance,
            taylor_cache_tailing_enhance_steps=taylor_cache_tailing_enhance_steps,
            taylor_cache_low_freqs_order=taylor_cache_low_freqs_order,
            taylor_cache_high_freqs_order=taylor_cache_high_freqs_order,
        )

        if cot_text:
            print(f"  Reasoning trace: {cot_text}")

        return circular_blend_edges(samples[0], blend_width)

    def __call__(self, image, **kwargs):
        return self.forward(image, **kwargs)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser("Commandline arguments for running HunyuanImage-3 panorama locally")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt", type=str, default=PANO_INSTRUCTION, help="Prompt to run")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of new tokens to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Use None for random seed.")
    parser.add_argument("--diff-infer-steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--height", type=int, default=960, help="Height of the generated panorama image.")
    parser.add_argument("--width", type=int, default=1952, help="Width of the generated panorama image.")
    parser.add_argument(
        "--use-system-prompt",
        type=str,
        choices=["None", "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "en_unified", "custom"],
        default="en_unified",
        help=(
            "Use system prompt. 'None' means no system prompt; 'dynamic' means "
            "the system prompt is determined by --bot-task; 'en_vanilla', "
            "'en_recaption', 'en_think_recaption' and 'en_unified' are four "
            "predefined system prompts; 'custom' means using the custom system "
            "prompt. When using 'custom', --system-prompt must be provided. "
            "Default to load from the model generation config."
        )
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt. Used when --use-system-prompt is 'custom'."
    )
    parser.add_argument(
        "--bot-task",
        type=str,
        choices=["image", "auto", "recaption", "think_recaption"],
        default="think_recaption",
        help=(
            "Type of task for the model. 'image' for direct image generation; "
            "'auto' for text generation; 'recaption' for re-write->image; "
            "'think_recaption' for think->re-write->image. "
            "Default to load from the model generation config."
        )
    )
    parser.add_argument("--verbose", type=int, default=2, help="Verbose level")
    parser.add_argument("--blend-width", type=int, default=32, help="Edge blending width for seamless panorama.")
    parser.add_argument(
        "--infer-align-image-size",
        action="store_true",
        help="Whether to align the target image size to the src image size."
    )
    # ======================== Taylor Cache ========================
    parser.add_argument("--use-taylor-cache", action="store_true", help="Use Taylor Cache when sampling.")
    parser.add_argument("--taylor-cache-interval", type=int, default=5, help="Interval of Taylor Cache.")
    parser.add_argument("--taylor-cache-order", type=int, default=2, help="Order of Taylor Cache.")
    parser.add_argument(
        "--taylor-cache-enable-first-enhance",
        action="store_true",
        help="Enable first enhance when using Taylor Cache."
    )
    parser.add_argument(
        "--taylor-cache-first-enhance-steps",
        type=int,
        default=3,
        help="First enhance steps when using Taylor Cache (>2)."
    )
    parser.add_argument(
        "--taylor-cache-enable-tailing-enhance",
        action="store_true",
        help="Enable tailing enhance when using Taylor Cache."
    )
    parser.add_argument(
        "--taylor-cache-tailing-enhance-steps",
        type=int,
        default=1,
        help="Tailing enhance steps when using Taylor Cache."
    )
    parser.add_argument(
        "--taylor-cache-low-freqs-order",
        type=int,
        default=2,
        help="Low freqs order when using Taylor Cache."
    )
    parser.add_argument(
        "--taylor-cache-high-freqs-order",
        type=int,
        default=2,
        help="High freqs order when using Taylor Cache."
    )

    parser.add_argument("--pretrained-model-name-or-path", type=str,
                        default=HunyuanPanoPipeline.DEFAULT_MODEL_ID,
                        help="HuggingFace repo ID or local path to the model")
    parser.add_argument("--subfolder", type=str,
                        default=HunyuanPanoPipeline.DEFAULT_SUBFOLDER,
                        help="Subfolder inside the repo containing the model weights (default: HY-Pano-2.0)")
    parser.add_argument("--attn-impl", type=str, default="sdpa", choices=["sdpa", "flash_attention_2"],
                        help="Attention implementation. 'flash_attention_2' requires flash attention to be installed.")
    parser.add_argument("--moe-impl", type=str, default="eager", choices=["eager", "flashinfer"],
                        help="MoE implementation. 'flashinfer' requires FlashInfer to be installed.")

    parser.add_argument("--save", type=str, default=None,
                        help="Path to save the generated image (default: <input_stem>_panorama.png)")
    parser.add_argument("--reproduce", action="store_true", help="Whether to reproduce the results")

    return parser.parse_args()


def main(args):
    # Reproducibility
    if args.reproduce:
        set_reproducibility(True, global_seed=args.seed)

    # Build pipeline
    pipeline = HunyuanPanoPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder=args.subfolder,
        attn_impl=args.attn_impl,
        moe_impl=args.moe_impl,
    )

    # Run inference
    output = pipeline(
        args.image,
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        use_system_prompt=args.use_system_prompt,
        system_prompt=args.system_prompt,
        bot_task=args.bot_task,
        diff_infer_steps=args.diff_infer_steps,
        verbose=args.verbose,
        max_new_tokens=args.max_new_tokens,
        infer_align_image_size=args.infer_align_image_size,
        blend_width=args.blend_width,
        use_taylor_cache=args.use_taylor_cache,
        taylor_cache_interval=args.taylor_cache_interval,
        taylor_cache_order=args.taylor_cache_order,
        taylor_cache_enable_first_enhance=args.taylor_cache_enable_first_enhance,
        taylor_cache_first_enhance_steps=args.taylor_cache_first_enhance_steps,
        taylor_cache_enable_tailing_enhance=args.taylor_cache_enable_tailing_enhance,
        taylor_cache_tailing_enhance_steps=args.taylor_cache_tailing_enhance_steps,
        taylor_cache_low_freqs_order=args.taylor_cache_low_freqs_order,
        taylor_cache_high_freqs_order=args.taylor_cache_high_freqs_order,
    )

    # Save
    save_path = args.save
    if save_path is None:
        input_stem = Path(args.image).stem
        save_path = f"{input_stem}_panorama.png"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    output.save(save_path)
    print(f"Image saved to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
