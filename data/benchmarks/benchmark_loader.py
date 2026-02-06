import json
from pathlib import Path
from typing import Dict, List, Union, Optional

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    hf_hub_download = None
    snapshot_download = None


class BenchmarkLoader:
    """
    Universal benchmark loader.
    Accepts externally registered data_info and loads test cases
    from local paths or HuggingFace repos.
    """

    SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"}

    def __init__(self):
        pass

    def load_benchmark(
        self,
        task_type: str,
        benchmark_name: str,
        data_path: Optional[Union[str, Path]] = None,
        data_info: Optional[Dict] = None,
        local_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Load benchmark test cases.

        Args:
            task_type:      Task category, e.g. "navigation".
            benchmark_name: Name of the benchmark, e.g. "sf_nav_vidgen_test".
            data_path:      Local directory or HuggingFace repo id.
            data_info:      dict describing how to read the benchmark.
                            Must contain "input_keys", "output_keys",
                            "metadata_path".
            local_dir:      Download destination for HuggingFace data.
                            If None, defaults to
                            ~/.cache/sceneflow/benchmarks/<repo>.
            **kwargs:       Extra arguments forwarded to HuggingFace
                            download (e.g. token, revision).

        Returns:
            List of dicts, one per test case. Media file values are
            replaced by absolute paths.
        """
        if data_info is None:
            raise ValueError(
                "data_info must be provided. It should contain at least "
                "'input_keys', 'output_keys', and 'metadata_path'."
            )
        self._validate_data_info(data_info)

        # 1. resolve base path (local / HuggingFace)
        base_path = self._resolve_data_path(
            data_path, local_dir=local_dir, **kwargs
        )

        # 2. load metadata
        metadata_file = base_path / data_info["metadata_path"]
        metadata_entries = self._load_metadata(metadata_file)

        # 3. assemble test cases
        test_cases = self._assemble_test_cases(
            metadata_entries=metadata_entries,
            input_keys=data_info["input_keys"],
            perception_data_path=data_info.get("perception_data_path", ""),
            base_path=base_path,
        )

        return test_cases

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------
    @staticmethod
    def _validate_data_info(data_info: Dict):
        required = ["input_keys", "output_keys", "metadata_path"]
        missing = [k for k in required if k not in data_info]
        if missing:
            raise ValueError(
                f"data_info is missing required keys: {missing}"
            )

    def _resolve_data_path(
        self,
        data_path: Optional[Union[str, Path]],
        local_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Path:
        if data_path is None:
            raise ValueError("data_path must be provided.")

        local = Path(data_path)
        if local.exists():
            return local.resolve()

        return download_from_hf(
            repo_id=str(data_path),
            local_dir=local_dir,
            **kwargs
        )

    def _load_metadata(self, metadata_file: Path) -> List[Dict]:
        metadata_file = Path(metadata_file)
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}"
            )

        suffix = metadata_file.suffix.lower()

        if suffix == ".jsonl":
            entries: List[Dict] = []
            with open(metadata_file, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Invalid JSON on line {line_no} of "
                            f"{metadata_file}: {e}"
                        )
            return entries

        if suffix == ".json":
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]

        raise ValueError(
            f"Unsupported metadata format '{suffix}'. Use .json or .jsonl"
        )

    def _is_media_file(self, value: str) -> bool:
        suffix = Path(value).suffix.lower()
        return suffix in (
            self.SUPPORTED_IMAGE_EXTENSIONS | self.SUPPORTED_VIDEO_EXTENSIONS
        )

    def _assemble_test_cases(
        self,
        metadata_entries: List[Dict],
        input_keys: List[str],
        perception_data_path: str,
        base_path: Path,
    ) -> List[Dict]:
        """
        For every metadata entry build a test-case dict.
        String values that look like media filenames are expanded to
        absolute paths: base_path / perception_data_path / filename.
        Extra keys beyond input_keys are preserved.
        """
        media_base = base_path / perception_data_path

        test_cases: List[Dict] = []
        for idx, entry in enumerate(metadata_entries):
            missing = [k for k in input_keys if k not in entry]
            if missing:
                raise KeyError(
                    f"Metadata entry #{idx} is missing required "
                    f"input_keys: {missing}. Entry: {entry}"
                )

            test_case: Dict = {}
            for key, value in entry.items():
                if isinstance(value, str) and self._is_media_file(value):
                    test_case[key] = str((media_base / value).resolve())
                else:
                    test_case[key] = value

            test_cases.append(test_case)

        return test_cases


# ============================================================
# Standalone helpers
# ============================================================
def load_json_file(file_path: Union[str, Path]) -> Union[Dict, List]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def download_from_hf(
    repo_id: str,
    filename: Optional[str] = None,
    repo_type: str = "dataset",
    local_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> Path:
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required. "
            "Install it with: pip install huggingface-hub"
        )

    if local_dir is None:
        local_dir = (
            Path.home()
            / ".cache"
            / "sceneflow"
            / "benchmarks"
            / repo_id.replace("/", "_")
        )
    local_dir = Path(local_dir)

    if filename:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=str(local_dir),
            **kwargs,
        )
    else:
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            **kwargs,
        )

    return Path(downloaded_path)
