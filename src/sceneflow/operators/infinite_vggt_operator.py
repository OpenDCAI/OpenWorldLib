import os
import cv2
import numpy as np
import torch
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

from .base_operator import BaseOperator


class InfiniteVGGTOperator(BaseOperator):
    """Operator for InfiniteVGGT pipeline: interaction template and perception (image/video loading)."""

    def __init__(
        self,
        operation_types=["visual_instruction", "action_instruction"],
        interaction_template=[
            "export_ply",
            "export_glb",
            "export_depth",
            "point_cloud",
            "depth_map",
            "move_left",
            "move_right",
            "move_up",
            "move_down",
            "zoom_in",
            "zoom_out",
        ],
    ):
        super(InfiniteVGGTOperator, self).__init__(operation_types=operation_types)
        self.interaction_template = interaction_template
        self.interaction_template_init()

    def collect_paths(self, path: Union[str, Path]) -> List[str]:
        """Collect file paths from a file, directory, or txt list."""
        path = str(path)
        if os.path.isfile(path):
            if path.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    files = [line.strip() for line in f.readlines() if line.strip()]
            else:
                files = [path]
        else:
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            files = [
                os.path.join(path, n)
                for n in os.listdir(path)
                if not n.startswith(".") and os.path.splitext(n)[1].lower() in exts
            ]
            files.sort()
        return files

    def process_perception(
        self,
        input_signal: Union[str, np.ndarray, torch.Tensor, List[str]],
    ) -> Union[np.ndarray, List[np.ndarray], List[str]]:
        """
        Process visual signal: load image(s) or video frames.
        Video/image/audio loading is done here; do not put file loading in process_interaction.
        """
        if isinstance(input_signal, (str, Path)):
            path = str(input_signal)
            if os.path.isdir(path) or (path.lower().endswith(".txt")):
                image_paths = self.collect_paths(path)
                if not image_paths:
                    raise ValueError(f"No images found: {path}")
                return image_paths
            if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                frames = []
                cap = cv2.VideoCapture(path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                if not frames:
                    raise ValueError(f"No frames read from video: {path}")
                return frames
            raw = cv2.imread(path)
            if raw is None:
                raise ValueError(f"Could not read image: {path}")
            return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        if isinstance(input_signal, list):
            if all(isinstance(x, str) for x in input_signal):
                return input_signal
            return list(input_signal)
        if isinstance(input_signal, np.ndarray):
            if input_signal.max() > 1.0:
                input_signal = (input_signal / 255.0).astype(np.float32)
            return input_signal
        if isinstance(input_signal, torch.Tensor):
            arr = input_signal.permute(1, 2, 0).cpu().numpy() if input_signal.dim() == 3 else input_signal[0].permute(1, 2, 0).cpu().numpy()
            if arr.max() > 1.0:
                arr = arr / 255.0
            return arr
        raise ValueError(f"Unsupported input type: {type(input_signal)}")

    def check_interaction(self, interaction):
        """Check if interaction is in self.interaction_template."""
        if interaction not in self.interaction_template:
            raise ValueError(
                f"'{interaction}' not in interaction_template. "
                f"Available: {self.interaction_template}"
            )
        return True

    def get_interaction(self, interaction):
        """Add interaction after check_interaction; append to self.current_interaction."""
        self.check_interaction(interaction)
        self.current_interaction.append(interaction)

    def process_interaction(self, num_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Turn self.current_interaction into a form usable by representation/synthesis.
        Only handles text or camera controls (e.g. export_ply, export_glb, move_left).
        """
        if not self.current_interaction:
            raise ValueError("No interaction to process. Use get_interaction() first.")
        latest = self.current_interaction[-1]
        self.interaction_history.append(latest)
        out = {
            "output_format": "ply",
            "export_ply": False,
            "export_glb": False,
            "export_depth": False,
        }
        if latest == "export_ply":
            out["output_format"] = "ply"
            out["export_ply"] = True
        elif latest == "export_glb":
            out["output_format"] = "glb"
            out["export_glb"] = True
        elif latest == "export_depth":
            out["output_format"] = "depth"
            out["export_depth"] = True
        elif latest == "point_cloud":
            out["output_format"] = "ply"
            out["export_ply"] = True
        elif latest == "depth_map":
            out["output_format"] = "depth"
            out["export_depth"] = True
        elif latest in ("move_left", "move_right", "move_up", "move_down", "zoom_in", "zoom_out"):
            out["camera_control"] = latest
        if num_frames is not None:
            out["num_frames"] = num_frames
        return out

    def delete_last_interaction(self):
        """Remove the last item from current_interaction."""
        if not self.current_interaction:
            raise ValueError("No interaction to delete.")
        self.current_interaction = self.current_interaction[:-1]
