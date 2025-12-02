from pathlib import Path
import torch
from mirror_src.models.models.worldmirror import WorldMirror
from mirror_src.utils.inference_utils import extract_load_and_preprocess_images

# --- Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = WorldMirror.from_pretrained("/opt/tiger/WMflow/ckpts", local_files_only=True).to(device)
model.eval()

# --- Load Data ---
# 加载 N 张图像序列到张量
inputs = {}
inputs['img'] = extract_load_and_preprocess_images(
    Path("hunyuan_mirror/examples/realistic/Room_Cat"), # 视频或包含图像的目录
    fps=1, # 从视频提取帧的帧率
    target_size=518
).to(device)  # [1,N,3,H,W], 范围 [0,1]
# -- 加载先验（可选） --
# 配置条件标志和先验路径
cond_flags = [0, 0, 0]  # [camera_pose, depth, intrinsics]
prior_data = {
    'camera_pose': None,      # 相机位姿张量 [1, N, 4, 4]
    'depthmap': None,         # 深度图张量 [1, N, H, W]
    'camera_intrinsics': None # 相机内参张量 [1, N, 3, 3]
}
for idx, (key, data) in enumerate(prior_data.items()):
    if data is not None:
        cond_flags[idx] = 1
        inputs[key] = data

# --- Inference ---
with torch.no_grad():
    predictions = model(views=inputs, cond_flags=cond_flags)
torch.save(predictions, "predictions.pt")
print("done")