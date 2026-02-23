import sys
from pathlib import Path
sys.path.append("..")

from sceneflow.pipelines.infinite_vggt.pipeline_infinite_vggt import InfiniteVGGTPipeline

# Configure before running: HuggingFace repo or local dir with .pth
MODEL_PATH = "lch01/StreamVGGT"
DATA_PATH = "./data/test_case/test_image_seq_case1" #"/YOUR/IMAGE/DIR/OR/VIDEO/PATH"
OUTPUT_DIR = "./output/infinite_vggt"

# Optional: set interaction (export_ply / export_glb / export_depth)
INTERACTION = None  # or e.g. "export_glb"

pipeline = InfiniteVGGTPipeline.from_pretrained(
    pretrained_model_path=MODEL_PATH,
)

if INTERACTION is not None:
    pipeline.operator.get_interaction(INTERACTION)

results = pipeline(DATA_PATH)
results.save(OUTPUT_DIR)

# Note: 记录中需要安装 viser以及scikit-learn，但是viser好像没遇到需要安装，同时scikit-learn貌似安装其他依赖所以也装上了
# Note: 同时有些问题，就是最后只保留了pointcloud，但是用户想查看的话，还是需要渲染的，所以这里除了重建，还需要加上渲染的内容
