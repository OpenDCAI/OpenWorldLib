import os
from typing import List, Dict, Union, Any
from huggingface_hub import snapshot_download, hf_hub_download
import torch


benchmark_info_map = {}

# cache path for HF
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", None)

def load_benchmark_data(data_path: str) -> Dict[str, Any]:
    """
    load corresponding info from specific tasks
    :param data_path: str data oath:
                      1. 本地绝对路径：/xxx/benchmarks/task1_data
                      2. 本地相对路径：./task2_data 或 benchmarks/task3_data
                      3. HF仓库路径：username/task-benchmark 或 username/task-benchmark:main
    :return: standard data info
    """
    # 初始化返回结果
    result = {
        "data_path": data_path,  # 原始传入路径
        "data_type": None,       # local / hf
        "local_real_path": None, # 最终实际的本地路径（HF会下载到缓存，也会填充此值）
        "data": None,            # 核心：加载后的benchmark原始数据
        "meta": {}               # 附加元信息（如数据量、样本格式等）
    }

    # -------------------------- load from local path --------------------------
    if os.path.exists(data_path):
        result["data_type"] = "local"
        result["local_real_path"] = os.path.abspath(data_path)
        # 校验本地路径是文件/文件夹
        if not (os.path.isfile(data_path) or os.path.isdir(data_path)):
            raise FileNotFoundError(f"本地路径无效：{data_path}，非文件/文件夹")
        
        # 【核心】加载本地数据：请根据你的实际数据格式修改此部分！
        # 示例：支持读取文件夹下的所有样本文件/单一json/csv/txt，此处为通用示例
        if os.path.isdir(data_path):
            # 示例：遍历文件夹获取所有样本文件名（可替换为读取json/csv等）
            sample_files = [f for f in os.listdir(data_path) if not f.startswith(".")]
            sample_files.sort()  # 排序保证一致性
            result["data"] = sample_files
            result["meta"]["sample_num"] = len(sample_files)
            result["meta"]["sample_type"] = "dir_files"
        else:
            # 示例：读取单一文件（以json为例，可替换为csv/txt/pickle等）
            import json
            with open(data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            result["data"] = raw_data
            result["meta"]["sample_num"] = len(raw_data) if isinstance(raw_data, (list, dict)) else 1
            result["meta"]["sample_type"] = os.path.splitext(data_path)[-1]
    
    # -------------------------- load from huggingface position --------------------------
    else:
        # 判定为HF路径：格式为 用户名/仓库名 或 用户名/仓库名:分支
        try:
            result["data_type"] = "hf"
            # 下载HF仓库到本地缓存，返回缓存的本地绝对路径
            local_cache = snapshot_download(
                repo_id=data_path,
                cache_dir=HF_CACHE_DIR,
                resume_download=True,  # 断点续传
                ignore_patterns=[".git*", "README.md"]  # 忽略无关文件
            )
            result["local_real_path"] = local_cache

            # 【核心】加载HF缓存中的数据：与本地加载逻辑一致，保证下游使用无感知
            sample_files = [f for f in os.listdir(local_cache) if not f.startswith(".")]
            sample_files.sort()
            result["data"] = sample_files
            result["meta"]["sample_num"] = len(sample_files)
            result["meta"]["hf_repo"] = data_path
            result["meta"]["sample_type"] = "hf_dir_files"

        except Exception as e:
            raise RuntimeError(f"HF路径加载失败：{data_path}，错误信息：{str(e)}") \
                from e

    return result


def load_batch_benchmarks(path_list: List[str]) -> Dict[str, Any]:
    """批量加载多个benchmark数据"""
    batch_result = {}
    for idx, path in enumerate(path_list):
        batch_result[f"benchmark_{idx+1}"] = load_benchmark_data(path)
    return batch_result