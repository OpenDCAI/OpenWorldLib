from typing import Union, Dict


def _resolve_path(model_path: Union[str, Dict], key: str) -> str:
    """
    Parse the specified path from model_path.
        - If `model_path` is a str, return it directly (backward compatibility for single-path model).
        - If `model_path` is a dict, retrieve the value by key; raise a clear error if the key does not exist.
    """
    if isinstance(model_path, dict):
        if key not in model_path:
            raise KeyError(
                f"Expected key '{key}' in model_path dict, "
                f"but only found: {list(model_path.keys())}"
            )
        return model_path[key]
    return model_path


def load_matrix_game2_pipeline(model_path: Union[str, Dict], device: str):
    from sceneflow.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline
    return MatrixGame2Pipeline.from_pretrained(
        model_path=_resolve_path(model_path, "pretrained_model_path"),
        mode="universal",
        device=device,
    )


def load_hunyuan_game_craft_pipeline(model_path: Union[str, Dict], device: str):
    from sceneflow.pipelines.hunyuan_world.pipeline_hunyuan_game_craft import HunyuanGameCraftPipeline
    return HunyuanGameCraftPipeline.from_pretrained(
        model_path=_resolve_path(model_path, "pretrained_model_path"),
        device=device,
    )


def load_qwen2p5_omni_pipeline(model_path: Union[str, Dict], device: str):
    from sceneflow.pipelines.qwen.pipeline_qwen2p5_omni import Qwen2p5OmniPipeline
    return Qwen2p5OmniPipeline.from_pretrained(
        pretrained_model_path=_resolve_path(model_path, "pretrained_model_path"),
        use_audio_in_video=False,
        device=device,
    )


def load_wan2p2_pipeline(model_path: Union[str, Dict], device: str):
    from sceneflow.pipelines.wan.pipeline_wan_2p2 import Wan2p2Pipeline
    return Wan2p2Pipeline.from_pretrained(
        synthesis_model_path=_resolve_path(model_path, "pretrained_model_path"),
        task="ti2v-5B",
    )


def load_spirit_v1p5_pipeline(model_path: Union[str, Dict], device: str, norm_stats_path: str = None):
    from sceneflow.pipelines.spirit_ai.pipeline_spirit_v1p5 import SpiritV1p5Pipeline
    return SpiritV1p5Pipeline.from_pretrained(
        pretrained_model_path=_resolve_path(model_path, "pretrained_model_path"),
        norm_stats_path=norm_stats_path,
        device=device,
        use_bf16=True,
    )


## utilize lazy loader to load different tasks pipeline
video_gen_pipe = {
    "matrix-game2": load_matrix_game2_pipeline,
    "wan2p2": load_wan2p2_pipeline,
    "hunyuan-game-craft": load_hunyuan_game_craft_pipeline,
}

reasoning_pipe = {
    "qwen2p5-omni": load_qwen2p5_omni_pipeline,
}

three_dim_pipe = {

}

vla_pipe = {
    "spirit-v1p5": load_spirit_v1p5_pipeline,
}