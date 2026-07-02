from _path_defaults import env_int, env_str, optional_model_path, python_bin
from openworldlib.pipelines.gamma_world.pipeline_gamma_world import GammaWorldPipeline


def main():
    pipe = GammaWorldPipeline.from_pretrained(
        model_path=optional_model_path("GAMMA_WORLD_CHECKPOINT", "Gamma-World/causal-few-step/model.safetensors"),
        vae=optional_model_path("GAMMA_WORLD_VAE", "Gamma-World/tokenizer.pth"),
        text_encoder=optional_model_path("GAMMA_WORLD_TEXT_ENCODER", "Cosmos-Reason1-7B"),
        python_bin=python_bin("GAMMA_WORLD_PYTHON"),
    )
    result = pipe(
        output_dir=env_str("GAMMA_WORLD_OUTPUT", "./output/gamma_world"),
        eval_dir=env_str("GAMMA_WORLD_EVAL_DIR", "") or None,
        mode=env_str("GAMMA_WORLD_MODE", "causal_few_step"),
        n_players=env_int("GAMMA_WORLD_N_PLAYERS", 2),
        max_eval_samples=env_int("GAMMA_WORLD_MAX_EVAL_SAMPLES", 1),
        num_frames=env_int("GAMMA_WORLD_NUM_FRAMES", 189),
        cuda_visible_devices=env_str("CUDA_VISIBLE_DEVICES", "0"),
        timeout=None,
    )
    print("\n".join(result["video_paths"]))


if __name__ == "__main__":
    main()
