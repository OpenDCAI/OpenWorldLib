"""
Benchmark data loader module.
Supports loading benchmark data from local paths or HuggingFace datasets.
"""

import json
from pathlib import Path
from typing import Dict, List, Union, Optional
import importlib

# Optional imports
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    hf_hub_download = None
    snapshot_download = None


class BenchmarkLoader:
    """
    Universal benchmark loader that can load different types of benchmarks
    based on task category and benchmark name.
    """
    
    def __init__(self):
        self.benchmark_registry = {}
        self._register_benchmarks()
    
    def _register_benchmarks(self):
        """Register all available benchmarks"""
        # Generation benchmarks
        self.benchmark_registry['navigation_video_generation'] = {
            'loader_module': 'benchmarks.generation.navigation_video_generation.benchmark_load',
            'benchmarks': {}  # Will be populated by the module
        }
        # Add more benchmark types as needed
    
    def load_benchmark(
        self,
        task_type: str,
        benchmark_name: str,
        data_path: Optional[Union[str, Path]] = None,
        hf_repo_id: Optional[str] = None,
        split: str = "test",
        **kwargs
    ) -> List[Dict]:
        """
        Load benchmark data from local path or HuggingFace.
        
        Args:
            task_type: Type of task (e.g., 'navigation_video_generation')
            benchmark_name: Name of the benchmark
            data_path: Local path to the benchmark data
            hf_repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
            split: Dataset split to load (default: 'test')
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            List of benchmark samples, each as a dictionary
        """
        
        if task_type not in self.benchmark_registry:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Get the loader module for this task type
        loader_info = self.benchmark_registry[task_type]
        loader_module_name = loader_info['loader_module']
        
        try:
            # Import the specific benchmark loader
            loader_module = importlib.import_module(loader_module_name)
            
            # Call the load function from the module
            if hasattr(loader_module, 'load_benchmark'):
                return loader_module.load_benchmark(
                    benchmark_name=benchmark_name,
                    data_path=data_path,
                    hf_repo_id=hf_repo_id,
                    split=split,
                    **kwargs
                )
            else:
                raise AttributeError(
                    f"Module {loader_module_name} does not have 'load_benchmark' function"
                )
                
        except ImportError as e:
            raise ImportError(
                f"Failed to import loader module {loader_module_name}: {e}"
            )
    
    def list_benchmarks(self, task_type: Optional[str] = None) -> Dict:
        """
        List all available benchmarks.
        
        Args:
            task_type: Optional task type filter
            
        Returns:
            Dictionary of available benchmarks by task type
        """
        if task_type:
            if task_type not in self.benchmark_registry:
                raise ValueError(f"Unknown task type: {task_type}")
            return {task_type: self.benchmark_registry[task_type]}
        
        return self.benchmark_registry


def load_json_file(file_path: Union[str, Path]) -> Union[Dict, List]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data (dict or list)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def download_from_hf(
    repo_id: str,
    filename: Optional[str] = None,
    repo_type: str = "dataset",
    local_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> Path:
    """
    Download data from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        filename: Specific file to download (if None, downloads entire repo)
        repo_type: Type of repository ('dataset' or 'model')
        local_dir: Local directory to save the data
        **kwargs: Additional arguments for hf_hub_download/snapshot_download
        
    Returns:
        Path to the downloaded data
    """
    
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required to download from HuggingFace. "
            "Install it with: pip install huggingface-hub"
        )
    
    if filename:
        # Download single file
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            **kwargs
        )
    else:
        # Download entire repository
        if local_dir is None:
            local_dir = Path.home() / ".cache" / "sceneflow" / "benchmarks" / repo_id.replace("/", "_")
        
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            **kwargs
        )
    
    return Path(downloaded_path)
