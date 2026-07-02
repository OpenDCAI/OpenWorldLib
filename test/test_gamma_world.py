import os

from openworldlib.pipelines.gamma_world.pipeline_gamma_world import GammaWorldPipeline


model_path = os.environ.get("GAMMA_WORLD_CHECKPOINT") or None
required_components = {
    "vae": os.environ.get("GAMMA_WORLD_VAE") or None,
    "text_encoder": os.environ.get("GAMMA_WORLD_TEXT_ENCODER") or None,
}

pipeline = GammaWorldPipeline.from_pretrained(
    model_path=model_path,
    required_components=required_components,
    python_bin=os.environ.get("GAMMA_WORLD_PYTHON", "python"),
)

result = pipeline(
    output_dir=os.environ.get("GAMMA_WORLD_OUTPUT", "./output/gamma_world"),
    eval_dir=os.environ.get("GAMMA_WORLD_EVAL_DIR") or None,
    mode=os.environ.get("GAMMA_WORLD_MODE", "causal_few_step"),
    n_players=int(os.environ.get("GAMMA_WORLD_N_PLAYERS", "2")),
    max_eval_samples=int(os.environ.get("GAMMA_WORLD_MAX_EVAL_SAMPLES", "1")),
    num_frames=int(os.environ.get("GAMMA_WORLD_NUM_FRAMES", "189")),
    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    timeout=None,
)

print("\n".join(result["video_paths"]))
