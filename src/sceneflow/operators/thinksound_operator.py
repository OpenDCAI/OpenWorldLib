import os
import csv
import subprocess
import shlex
import numpy as np
import torch
from pathlib import Path
from typing import Union, Dict, Any

from .base_operator import BaseOperator

class ThinkSoundOperator(BaseOperator):
    """
    ThinkSound 数据处理 Operator
    
    负责从原始视频开始的全部数据预处理：
    1) 视频格式化 + 时长计算
    2) caption / caption_cot CSV 生成
    3) 调用 extract_latents.py 产生 demo.npz
    4) 加载 demo.npz 并构造模型输入
    """
    
    def __init__(
        self, 
        video_dir: Union[str, Path] = "videos",
        cot_dir: Union[str, Path] = "cot_coarse",
        results_dir: Union[str, Path] = "results",
        scripts_dir: Union[str, Path] = ".",
        operation_types: list = None
    ):
        """
        初始化 ThinkSoundOperator
        
        Args:
            operation_types: 操作类型列表
        """
        if operation_types is None:
            operation_types = ["video_processing", "feature_extraction", "npz_loading"]
        super().__init__(operation_types=operation_types)
        # 记录到实例上，方便 pipeline 等组件保存 / 调试
        self.opration_types = operation_types
        
        self.video_dir = Path(video_dir)
        self.cot_dir = Path(cot_dir)
        self.results_dir = Path(results_dir)
        self.scripts_dir = Path(scripts_dir)

        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.cot_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def check_interaction(
        self,
        video_path: Union[str, Path],
        title: str,
        description: str,
    ) -> bool:
        """
        检查一次 ThinkSound 交互是否有效：
            - video_path 必须存在
            - title / description 必须为非空字符串
        """
        if not isinstance(video_path, (str, Path)):
            raise TypeError("video_path must be a str or Path")
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"video_path not found: {video_path}")

        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string")

        return True

    def get_interaction(
        self,
        video_path: Union[str, Path],
        title: str,
        description: str,
    ) -> Dict[str, Any]:
        """
        记录当前交互信息，便于上层 pipeline 或日志使用。

        为了与其它 Operator 保持一致：
        - 将每次交互 append 到 BaseOperator.current_interaction（list）
        - 返回本次交互的字典对象
        """
        if self.check_interaction(video_path, title, description):
            interaction: Dict[str, Any] = {
                "video_path": str(video_path),
                "title": title,
                "description": description,
            }
            self.current_interaction.append(interaction)
            return interaction
        raise RuntimeError("check_interaction failed but no exception was raised.")
    
    #  视频转换与时长计算 
    def _prepare_video(self, video_path: Union[str, Path]) -> tuple[Path, float]:
        """
        将输入视频转换为 demoshell 使用的 MP4，并返回时长
        """
        video_path = Path(video_path)
        temp_video = self.video_dir / "demo.mp4"
        
        if video_path.suffix.lower() != ".mp4":
            cmd = f'ffmpeg -y -i "{video_path}" -c:v libx264 -preset fast -c:a aac "{temp_video}"'
            subprocess.run(shlex.split(cmd), check=True)
        else:
            temp_video.write_bytes(video_path.read_bytes())
        
        duration_cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{temp_video}"'
        result = subprocess.run(shlex.split(duration_cmd), capture_output=True, text=True, check=True)
        duration_sec = float(result.stdout.strip())

        # 为了与 ThinkSound 原版推理完全对齐，时长相关的下游逻辑都使用「向下取整后的整数秒」
        # 原版脚本中 duration_sec 由 CLI 参数给定（如 8.0），不会携带小数尾巴。
        duration_sec_int = int(duration_sec)
        
        return temp_video, duration_sec_int
    
    # 写 caption / COT 
    def _write_cot(self, title: str, description: str) -> Path:
        csv_path = self.cot_dir / "cot.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "caption", "caption_cot"])
            writer.writerow(["demo", title, description.replace('"', "'")])
        return csv_path
    
    # 调用 extract_latents.py 
    def _run_feature_extraction(self, duration_sec: float, use_half: bool = False):
        # cmd = [
        #     "python",
        #     str(self.scripts_dir / "extract_latents.py"),
        #     "--duration_sec",
        #     str(int(duration_sec)),
        # ]
        cmd = ["python", "/data0/hdl/sceneflow/SceneFlow/src/sceneflow/synthesis/audio_generation/thinksound/ThinkSound/extract_latents.py", "--duration_sec", str(int(duration_sec))]
        if use_half:
            cmd.append("--use_half")
        subprocess.run(cmd, check=True)
    
    # 加载 demo.npz 
    def _load_npz(self, duration: float) -> tuple[torch.Tensor, Dict[str, Any]]:
        npz_path = self.results_dir / "demo.npz"
        if not npz_path.exists():
            raise ValueError(f"feature npz not found: {npz_path}")
        
        npz_data = np.load(npz_path, allow_pickle=True)
        data = {key: npz_data[key] for key in npz_data.files}
        
        for key in data.keys():
            if isinstance(data[key], np.ndarray) and np.issubdtype(data[key].dtype, np.number):
                data[key] = torch.from_numpy(data[key])
        
        # 这里使用与模型配置中一致的公式；由于上游已对 duration 做过取整，
        # 可以避免出现 172 vs 173 这类 off-by-one 的长度不一致问题。
        latent_length = round(44100/64/32 * duration)
        audio = torch.zeros((1, 64, latent_length), dtype=torch.float32)
        
        metadata = data.copy()
        metadata["video_exist"] = torch.tensor(True)
        metadata["path"] = str(npz_path)
        metadata["id"] = "demo"
        metadata["relpath"] = "demo.npz"
        
        return audio, metadata
    
    def process_perception(
        self, 
        video_path: Union[str, Path],
        title: str,
        description: str,
        use_half: bool = False,
        device: str = "cuda",
        **kwargs
    ) -> Dict[str, Any]:

        # 记录交互并写入 history
        interaction = self.get_interaction(video_path, title, description)
        self.interaction_history.append(interaction)

        # 从最后一次交互中读取参数（与传参内容一致，保持行为不变）
        last_interaction = self.current_interaction[-1]
        video_path = last_interaction["video_path"]
        title = last_interaction["title"]
        description = last_interaction["description"]

        # 下面逻辑保持原有功能：视频准备、写 COT、提取特征、加载 npz
        temp_video, duration_sec = self._prepare_video(video_path)
        self._write_cot(title, description)
        self._run_feature_extraction(duration_sec, use_half=use_half)
        audio, metadata = self._load_npz(duration_sec)
        
        audio = audio.to(device)
        for k, v in metadata.items():
            if isinstance(v, torch.Tensor):
                metadata[k] = v.to(device)
        
        processed_data = {
            "batch": (audio, (metadata,)),
            "duration": duration_sec,
            "id": metadata["id"],
        }
        
        # 可选：清理临时视频
        if temp_video.exists():
            temp_video.unlink()
        
        return processed_data