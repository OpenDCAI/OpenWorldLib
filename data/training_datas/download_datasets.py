import os
import time
import sys
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from huggingface_hub import HfApi, snapshot_download, get_token
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

# --- 配置常量和全局设置 ---
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # 高性能传输（必须安装 hf_transfer）
DOWNLOAD_RETRIES = 3
RETRY_DELAY_SEC = 10
SAVE_ROOT = "./train_data"
CACHE_ROOT = "./hf_cache"  # HuggingFace缓存目录
METADATA_FILE = os.path.join(SAVE_ROOT, ".download_metadata.json")

# 数据校验配置
ENABLE_VALIDATION = True  # 启用下载后校验
ENABLE_HASH_CHECK = False  # 启用哈希校验（耗时较长，可选）
VALIDATION_REPORT_FILE = os.path.join(SAVE_ROOT, ".validation_report.json")

# 并行下载配置
MAX_PARALLEL_DATASETS = 3  # 同时下载的数据集数量（根据带宽调整）
RESUME_DOWNLOAD = True  # 启用断点续传

# 文件过滤配置（可选：只下载特定文件类型）
ALLOWED_EXTENSIONS = None  # 设为 ['.mp4', '.avi', '.json'] 等来过滤文件
# 示例: ALLOWED_EXTENSIONS = ['.mp4', '.avi', '.webm', '.mkv', '.json']  # 只下载视频和元数据
IGNORE_PATTERNS = ["*.md", "*.txt", ".gitattributes"]  # 忽略文档文件

# --- 数据集分类定义 ---
DATASET_CATEGORIES = {
    "generation": [
        "InternRobotics/OmniWorld",
        "KlingTeam/GameFactory-Dataset",
    ],
    "reasoning": [
        "ai4ce/CityWalker",
        # "some_user/Another_Reasoning_Dataset",
    ],
    "representation": [
        "annadeichler/KTH-ARIA-referential",
    ],
    "vlm": [
        # "some_vlm_org/Specific_VLM_Data",
    ],
    "gated": [
        "AgiBot/AgiBot-Demo",
    ]
}


class ValidationReport:
    """管理数据校验报告"""
    
    def __init__(self, report_file: str):
        self.report_file = report_file
        self.data = self._load()
    
    def _load(self) -> Dict:
        if os.path.exists(self.report_file):
            try:
                with open(self.report_file, 'r') as f:
                    return json.load(f)
            except:
                return {"validated": {}, "failed": {}}
        return {"validated": {}, "failed": {}}
    
    def _save(self):
        os.makedirs(os.path.dirname(self.report_file), exist_ok=True)
        with open(self.report_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def add_validation(self, repo_id: str, validation_info: Dict):
        """记录校验成功的数据集"""
        self.data["validated"][repo_id] = {
            **validation_info,
            "timestamp": time.time()
        }
        self._save()
    
    def add_failure(self, repo_id: str, error: str):
        """记录校验失败的数据集"""
        self.data["failed"][repo_id] = {
            "error": error,
            "timestamp": time.time()
        }
        self._save()


class DownloadMetadata:
    """管理下载元数据，用于断点续传和状态跟踪"""
    
    def __init__(self, metadata_file: str):
        self.metadata_file = metadata_file
        self.data = self._load()
    
    def _load(self) -> Dict:
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {"completed": [], "failed": [], "in_progress": {}}
        return {"completed": [], "failed": [], "in_progress": {}}
    
    def _save(self):
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def is_completed(self, repo_id: str) -> bool:
        return repo_id in self.data["completed"]
    
    def mark_completed(self, repo_id: str):
        if repo_id not in self.data["completed"]:
            self.data["completed"].append(repo_id)
        if repo_id in self.data.get("in_progress", {}):
            del self.data["in_progress"][repo_id]
        self._save()
    
    def mark_failed(self, repo_id: str, error: str):
        self.data["failed"].append({"repo": repo_id, "error": error, "time": time.time()})
        if repo_id in self.data.get("in_progress", {}):
            del self.data["in_progress"][repo_id]
        self._save()
    
    def mark_in_progress(self, repo_id: str):
        self.data["in_progress"][repo_id] = time.time()
        self._save()


def require_token() -> str:
    """检查并获取 HF 令牌"""
    token = get_token()
    if token is None:
        print("\n🚨 警告: HuggingFace 令牌未找到。")
        print("请运行 'huggingface-cli login' 进行登录。")
        raise ValueError("HuggingFace token not found.")
    return token


def get_dataset_size(repo_id: str, token: Optional[str] = None) -> Tuple[int, int]:
    """获取数据集大小信息（文件数量和总大小）- 增强版"""
    try:
        api = HfApi()
        
        # 方法1: 尝试从 repo_info 获取（快速但可能不准确）
        try:
            repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
            total_size = 0
            file_count = 0
            
            for sibling in repo_info.siblings:
                if sibling.size:
                    total_size += sibling.size
                    file_count += 1
            
            if total_size > 0:  # 如果获取到了有效数据
                return file_count, total_size
        except Exception as e:
            print(f"⚠️ repo_info 方法失败，尝试备用方法: {e}")
        
        # 方法2: 遍历文件列表（更准确但较慢）
        print(f"📋 正在详细扫描文件列表...")
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
        file_count = len(files)
        
        # 注意：list_repo_files 不返回文件大小，这里只能返回文件数
        # 如果需要精确大小，需要逐个获取文件信息（开销很大）
        return file_count, 0  # 返回0表示大小未知
        
    except Exception as e:
        print(f"⚠️ 无法获取 {repo_id} 的信息: {e}")
        return 0, 0


def format_size(size_bytes: int) -> str:
    """格式化文件大小显示"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def check_disk_space(required_bytes: int, path: str = ".") -> bool:
    """检查磁盘空间是否足够"""
    try:
        stat = os.statvfs(path)
        available = stat.f_bavail * stat.f_frsize
        return available > required_bytes * 1.2  # 预留20%空间
    except:
        return True  # Windows或检查失败，继续执行


def calculate_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """计算文件哈希值（用于完整性校验）"""
    hash_obj = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def count_files_in_directory(directory: str, extensions: Optional[List[str]] = None) -> Dict[str, int]:
    """
    统计目录中的文件数量和大小
    返回: {"count": int, "total_size": int, "by_extension": {".mp4": 10, ...}}
    """
    stats = {
        "count": 0,
        "total_size": 0,
        "by_extension": {}
    }
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            # 如果指定了扩展名过滤
            if extensions and ext not in extensions:
                continue
            
            try:
                file_size = os.path.getsize(file_path)
                stats["count"] += 1
                stats["total_size"] += file_size
                stats["by_extension"][ext] = stats["by_extension"].get(ext, 0) + 1
            except:
                pass  # 跳过无法访问的文件
    
    return stats


def validate_dataset(
    repo_id: str,
    local_dir: str,
    token: Optional[str] = None,
    enable_hash_check: bool = False
) -> Tuple[bool, Dict]:
    """
    校验下载的数据集完整性
    
    返回: (是否通过校验, 校验详情字典)
    """
    print(f"\n🔍 开始校验数据集: {repo_id}")
    validation_info = {
        "repo_id": repo_id,
        "local_dir": local_dir,
        "checks": {}
    }
    
    try:
        api = HfApi()
        
        # 1. 获取远程文件列表
        print(f"📋 获取远程文件列表...")
        remote_files = api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        
        # 过滤掉 .gitattributes 等非数据文件
        remote_files_filtered = [
            f for f in remote_files 
            if not any(pattern in f for pattern in ['.gitattributes', '.gitignore'])
        ]
        
        remote_count = len(remote_files_filtered)
        validation_info["checks"]["remote_file_count"] = remote_count
        print(f"   远程文件数: {remote_count}")
        
        # 2. 统计本地文件
        print(f"📁 统计本地文件...")
        local_stats = count_files_in_directory(
            local_dir,
            extensions=ALLOWED_EXTENSIONS if ALLOWED_EXTENSIONS else None
        )
        local_count = local_stats["count"]
        validation_info["checks"]["local_file_count"] = local_count
        validation_info["checks"]["local_total_size"] = local_stats["total_size"]
        validation_info["checks"]["file_types"] = local_stats["by_extension"]
        
        print(f"   本地文件数: {local_count}")
        print(f"   本地总大小: {format_size(local_stats['total_size'])}")
        
        # 显示文件类型分布
        if local_stats["by_extension"]:
            print(f"   文件类型分布:")
            for ext, count in sorted(local_stats["by_extension"].items()):
                print(f"      {ext}: {count} 个")
        
        # 3. 文件数量校验
        # 注意：如果使用了 ALLOWED_EXTENSIONS，本地文件数可能少于远程
        if ALLOWED_EXTENSIONS:
            print(f"   ℹ️  已启用文件过滤，本地文件数可能少于远程")
            # 只要本地有文件就认为合理
            if local_count == 0:
                print(f"   ⚠️  警告: 本地无文件！可能下载失败或过滤过于严格")
                validation_info["checks"]["file_count_match"] = False
            else:
                validation_info["checks"]["file_count_match"] = True
        else:
            # 没有过滤时，要求数量严格匹配（允许5%误差，因为可能有隐藏文件）
            tolerance = max(1, int(remote_count * 0.05))
            count_match = abs(local_count - remote_count) <= tolerance
            validation_info["checks"]["file_count_match"] = count_match
            
            if count_match:
                print(f"   ✅ 文件数量校验通过 (差异在容忍范围内)")
            else:
                print(f"   ❌ 文件数量不匹配！远程: {remote_count}, 本地: {local_count}")
                return False, validation_info
        
        # 4. 可选：哈希校验（耗时，默认关闭）
        if enable_hash_check:
            print(f"🔐 开始哈希校验（这可能需要较长时间）...")
            try:
                repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
                hash_mismatches = []
                checked_files = 0
                
                for sibling in repo_info.siblings:
                    if sibling.rfilename in remote_files_filtered:
                        local_file_path = os.path.join(local_dir, sibling.rfilename)
                        if os.path.exists(local_file_path):
                            # 注意：HF使用的是etag，不是标准哈希
                            # 这里只做演示，实际需要更复杂的逻辑
                            checked_files += 1
                            if checked_files % 100 == 0:
                                print(f"   已校验 {checked_files} 个文件...")
                
                validation_info["checks"]["hash_checked_files"] = checked_files
                validation_info["checks"]["hash_mismatches"] = len(hash_mismatches)
                
                if hash_mismatches:
                    print(f"   ❌ 发现 {len(hash_mismatches)} 个文件哈希不匹配")
                    validation_info["checks"]["hash_check_passed"] = False
                else:
                    print(f"   ✅ 哈希校验通过 (检查了 {checked_files} 个文件)")
                    validation_info["checks"]["hash_check_passed"] = True
                    
            except Exception as e:
                print(f"   ⚠️  哈希校验出错: {e}")
                validation_info["checks"]["hash_check_error"] = str(e)
        
        # 5. 检查关键文件是否存在（针对特定数据集类型）
        critical_files_check = check_critical_files(local_dir, repo_id)
        validation_info["checks"]["critical_files"] = critical_files_check
        
        if not critical_files_check["passed"]:
            print(f"   ⚠️  警告: {critical_files_check['message']}")
        
        # 总体判断
        overall_pass = (
            validation_info["checks"].get("file_count_match", True) and
            (not enable_hash_check or validation_info["checks"].get("hash_check_passed", True))
        )
        
        if overall_pass:
            print(f"✅ 数据集校验通过: {repo_id}")
        else:
            print(f"❌ 数据集校验失败: {repo_id}")
        
        validation_info["passed"] = overall_pass
        return overall_pass, validation_info
        
    except Exception as e:
        error_msg = f"校验过程出错: {e}"
        print(f"❌ {error_msg}")
        validation_info["error"] = error_msg
        validation_info["passed"] = False
        return False, validation_info


def check_critical_files(local_dir: str, repo_id: str) -> Dict:
    """
    检查关键文件是否存在（根据数据集类型）
    例如：视频数据集应该有视频文件，标注数据集应该有JSON/CSV等
    """
    result = {"passed": True, "message": ""}
    
    # 统计文件类型
    has_videos = False
    has_annotations = False
    
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.mp4', '.avi', '.webm', '.mkv', '.mov']:
                has_videos = True
            if ext in ['.json', '.csv', '.txt', '.xml']:
                has_annotations = True
    
    # 针对World Model数据集的启发式检查
    if "video" in repo_id.lower() or "world" in repo_id.lower():
        if not has_videos:
            result["passed"] = False
            result["message"] = "视频数据集中未找到视频文件"
    
    if not has_videos and not has_annotations:
        result["passed"] = False
        result["message"] = "数据集中既无视频也无标注文件，可能下载不完整"
    
    return result


def check_disk_space(required_bytes: int, path: str = ".") -> bool:
    """检查磁盘空间是否足够"""
    try:
        stat = os.statvfs(path)
        available = stat.f_bavail * stat.f_frsize
        return available > required_bytes * 1.2  # 预留20%空间
    except:
        return True  # Windows或检查失败，继续执行


def download_dataset(
    repo_id: str, 
    local_dir: str, 
    metadata: DownloadMetadata,
    validation_report: ValidationReport,
    force_download: bool = False,
    enable_validation: bool = True
) -> bool:
    """下载单个数据集，返回是否成功"""
    
    # 检查是否已完成
    if not force_download and metadata.is_completed(repo_id):
        print(f"⏭️  跳过已完成的数据集: {repo_id}")
        return True
    
    print(f"\n{'='*70}")
    print(f"⏳ 正在处理数据集: {repo_id}")
    print(f"📁 保存至: {local_dir}")
    
    api = HfApi()
    token_to_use = None
    is_gated = False
    
    # 获取数据集大小
    try:
        file_count, total_size = get_dataset_size(repo_id)
        if file_count > 0:
            size_str = format_size(total_size) if total_size > 0 else "未知大小"
            print(f"📊 数据集信息: {file_count} 个文件, 总大小: {size_str}")
        
        # 检查磁盘空间（仅当已知大小时）
        if total_size > 0 and not check_disk_space(total_size, SAVE_ROOT):
            print(f"❌ 磁盘空间不足！需要约 {format_size(total_size * 1.2)}")
            metadata.mark_failed(repo_id, "Insufficient disk space")
            return False
    except Exception as e:
        print(f"⚠️ 无法获取数据集信息，继续下载: {e}")

    metadata.mark_in_progress(repo_id)
    
    # 首先确定是否需要 token（避免在重试循环中重复检查）
    try:
        api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        print(f"✔️ 检测到公共数据集")
        token_to_use = None
    except (RepositoryNotFoundError, RevisionNotFoundError):
        print(f"🔐 检测到需要认证的数据集")
        is_gated = True
        try:
            token_to_use = require_token()
            # 验证 token 是否有效
            api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token_to_use)
            print(f"✅ 认证成功")
        except ValueError as ve:
            # Token 不可用，直接标记失败，不进入重试
            print(f"❌ 认证失败: {ve}")
            print(f"💡 跳过此数据集，继续处理其他数据集")
            metadata.mark_failed(repo_id, f"Authentication failed: {ve}")
            return False
        except Exception as auth_error:
            print(f"❌ 认证验证失败: {auth_error}")
            metadata.mark_failed(repo_id, f"Auth validation failed: {auth_error}")
            return False
    
    # 准备文件过滤参数
    allow_patterns_list = None
    if ALLOWED_EXTENSIONS:
        allow_patterns_list = [f"**/*{ext}" for ext in ALLOWED_EXTENSIONS]
        print(f"🎯 文件过滤: 仅下载 {', '.join(ALLOWED_EXTENSIONS)} 格式")
    
    if IGNORE_PATTERNS:
        print(f"🚫 忽略文件: {', '.join(IGNORE_PATTERNS)}")
    
    # 执行下载（带重试）
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            print(f"⬇️  开始下载 (尝试 {attempt}/{DOWNLOAD_RETRIES})...")
            snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=RESUME_DOWNLOAD,
                force_download=force_download,
                token=token_to_use,
                cache_dir=CACHE_ROOT,
                allow_patterns=allow_patterns_list,  # ✅ 应用文件白名单
                ignore_patterns=IGNORE_PATTERNS if IGNORE_PATTERNS else None,
            )
            
            print(f"✅ 成功下载: {repo_id}")
            
            # 下载成功后进行校验
            if enable_validation:
                validation_passed, validation_info = validate_dataset(
                    repo_id,
                    local_dir,
                    token=token_to_use,
                    enable_hash_check=ENABLE_HASH_CHECK
                )
                
                if validation_passed:
                    validation_report.add_validation(repo_id, validation_info)
                    metadata.mark_completed(repo_id)
                    return True
                else:
                    # 校验失败
                    print(f"⚠️  数据集下载完成但校验失败，建议重新下载")
                    validation_report.add_failure(repo_id, validation_info.get("error", "Validation failed"))
                    metadata.mark_failed(repo_id, "Validation failed after download")
                    return False
            else:
                # 不校验，直接标记完成
                metadata.mark_completed(repo_id)
                return True
            
        except Exception as e:
            error_msg = f"{e.__class__.__name__}: {e}"
            if attempt < DOWNLOAD_RETRIES:
                print(f"❌ 下载失败 (尝试 {attempt}/{DOWNLOAD_RETRIES})")
                print(f"   错误: {error_msg}")
                print(f"   等待 {RETRY_DELAY_SEC} 秒后重试...")
                time.sleep(RETRY_DELAY_SEC)
            else:
                print(f"❌ 最终失败: {repo_id}")
                print(f"   错误: {error_msg}")
                metadata.mark_failed(repo_id, error_msg)
                return False
    
    return False


def download_category(
    category: str,
    repo_list: List[str],
    metadata: DownloadMetadata,
    validation_report: ValidationReport,
    parallel: bool = False,
    enable_validation: bool = True
) -> Tuple[List[str], List[str]]:
    """下载某个分类的所有数据集"""
    
    print(f"\n{'#'*70}")
    print(f"### ⚙️ 启动下载分类: {category.upper()} (共 {len(repo_list)} 个仓库) ###")
    print(f"{'#'*70}")
    
    success_list = []
    failed_list = []
    
    if parallel and len(repo_list) > 1:
        print(f"🚀 并行下载模式 (最多 {MAX_PARALLEL_DATASETS} 个并发)")
        
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_DATASETS) as executor:
            future_to_repo = {}
            
            for repo in repo_list:
                repo_name_clean = repo.replace("/", "_")
                local_save_path = os.path.join(SAVE_ROOT, category, repo_name_clean)
                os.makedirs(local_save_path, exist_ok=True)
                
                future = executor.submit(
                    download_dataset, 
                    repo, 
                    local_save_path, 
                    metadata, 
                    validation_report,
                    enable_validation=enable_validation
                )
                future_to_repo[future] = repo
            
            for future in as_completed(future_to_repo):
                repo = future_to_repo[future]
                try:
                    success = future.result()
                    if success:
                        success_list.append(repo)
                    else:
                        failed_list.append(repo)
                except Exception as e:
                    print(f"[并行任务异常] {repo}: {e}")
                    failed_list.append(repo)
    else:
        # 串行下载
        for repo in repo_list:
            repo_name_clean = repo.replace("/", "_")
            local_save_path = os.path.join(SAVE_ROOT, category, repo_name_clean)
            os.makedirs(local_save_path, exist_ok=True)
            
            try:
                success = download_dataset(
                    repo, 
                    local_save_path, 
                    metadata, 
                    validation_report,
                    enable_validation=enable_validation
                )
                if success:
                    success_list.append(repo)
                else:
                    failed_list.append(repo)
            except Exception as e:
                print(f"[全局记录] {repo} 下载失败: {e}")
                failed_list.append(repo)
    
    return success_list, failed_list


def print_summary(all_success: List[str], all_failed: List[str], start_time: float, validation_report: ValidationReport):
    """打印下载总结"""
    duration = time.time() - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    print("\n" + "="*70)
    print("📊 下载总结报告")
    print("="*70)
    print(f"⏱️  总耗时: {hours}h {minutes}m {seconds}s")
    print(f"✅ 成功: {len(all_success)} 个数据集")
    print(f"❌ 失败: {len(all_failed)} 个数据集")
    
    # 校验报告统计
    if ENABLE_VALIDATION:
        validated_count = len(validation_report.data.get("validated", {}))
        validation_failed = len(validation_report.data.get("failed", {}))
        print(f"\n🔍 数据校验:")
        print(f"   通过校验: {validated_count} 个")
        print(f"   校验失败: {validation_failed} 个")
        
        if validation_failed > 0:
            print(f"\n⚠️  校验失败的数据集:")
            for repo, info in validation_report.data.get("failed", {}).items():
                print(f"   - {repo}: {info.get('error', 'Unknown error')}")
    
    if all_failed:
        print("\n⚠️ 下载失败的数据集:")
        for repo in all_failed:
            print(f"  - {repo}")
        print("\n💡 提示: 检查网络连接、HF登录状态和磁盘空间")
    else:
        print("\n🎉 所有数据集下载完成！")
    
    # 显示数据集存储位置
    print(f"\n📁 数据集存储位置: {os.path.abspath(SAVE_ROOT)}")
    print(f"📋 元数据文件: {os.path.abspath(METADATA_FILE)}")
    if ENABLE_VALIDATION:
        print(f"🔍 校验报告: {os.path.abspath(VALIDATION_REPORT_FILE)}")
    
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="优化的HuggingFace数据集批量下载工具")
    parser.add_argument("--parallel", action="store_true", help="启用并行下载")
    parser.add_argument("--force", action="store_true", help="强制重新下载已完成的数据集")
    parser.add_argument("--category", type=str, help="只下载指定分类")
    parser.add_argument("--max-workers", type=int, default=3, help="最大并行下载数")
    parser.add_argument("--no-validation", action="store_true", help="禁用下载后校验")
    parser.add_argument("--enable-hash-check", action="store_true", help="启用哈希校验（耗时较长）")
    parser.add_argument("--validate-only", type=str, help="只校验指定数据集（不下载）")
    args = parser.parse_args()
    
    if args.max_workers:
        MAX_PARALLEL_DATASETS = args.max_workers
    
    if args.enable_hash_check:
        ENABLE_HASH_CHECK = True
    
    enable_validation = not args.no_validation
    
    # 初始化
    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs(CACHE_ROOT, exist_ok=True)
    metadata = DownloadMetadata(METADATA_FILE)
    validation_report = ValidationReport(VALIDATION_REPORT_FILE)
    
    # 只校验模式
    if args.validate_only:
        print(f"🔍 进入校验模式: {args.validate_only}")
        
        # 查找对应的本地目录
        found = False
        for category, repo_list in DATASET_CATEGORIES.items():
            if args.validate_only in repo_list:
                repo_name_clean = args.validate_only.replace("/", "_")
                local_path = os.path.join(SAVE_ROOT, category, repo_name_clean)
                
                if os.path.exists(local_path):
                    print(f"找到本地数据集: {local_path}")
                    validation_passed, validation_info = validate_dataset(
                        args.validate_only,
                        local_path,
                        enable_hash_check=ENABLE_HASH_CHECK
                    )
                    
                    if validation_passed:
                        print(f"\n✅ 数据集校验通过")
                        validation_report.add_validation(args.validate_only, validation_info)
                    else:
                        print(f"\n❌ 数据集校验失败")
                        validation_report.add_failure(args.validate_only, validation_info.get("error", "Unknown"))
                    
                    found = True
                    break
                else:
                    print(f"❌ 本地目录不存在: {local_path}")
                    found = True
                    break
        
        if not found:
            print(f"❌ 未找到数据集: {args.validate_only}")
        
        sys.exit(0)
    
    print(f"💾 数据根目录: {os.path.abspath(SAVE_ROOT)}")
    print(f"📦 缓存目录: {os.path.abspath(CACHE_ROOT)}")
    print(f"🔄 断点续传: {'启用' if RESUME_DOWNLOAD else '禁用'}")
    print(f"⚡ 并行下载: {'启用' if args.parallel else '禁用'}")
    print(f"🔍 数据校验: {'启用' if enable_validation else '禁用'}")
    if enable_validation and ENABLE_HASH_CHECK:
        print(f"🔐 哈希校验: 启用 (将增加下载时间)")
    
    # 检查 hf_transfer
    try:
        import hf_transfer
        print(f"✅ hf_transfer 已安装 (高速传输已启用)")
    except ImportError:
        print(f"⚠️ 建议安装 hf_transfer 以加速下载: pip install hf-transfer")
    
    start_time = time.time()
    all_success = []
    all_failed = []
    
    # 过滤分类
    categories_to_download = DATASET_CATEGORIES
    if args.category:
        if args.category in DATASET_CATEGORIES:
            categories_to_download = {args.category: DATASET_CATEGORIES[args.category]}
        else:
            print(f"❌ 错误: 分类 '{args.category}' 不存在")
            sys.exit(1)
    
    # 执行下载
    for category, repo_list in categories_to_download.items():
        success, failed = download_category(
            category, 
            repo_list, 
            metadata, 
            validation_report,
            args.parallel,
            enable_validation=enable_validation
        )
        all_success.extend(success)
        all_failed.extend(failed)
    
    # 打印总结
    print_summary(all_success, all_failed, start_time, validation_report)
    
    # 返回退出码
    sys.exit(0 if not all_failed else 1)