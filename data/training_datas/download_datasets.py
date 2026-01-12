import os
import time
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, snapshot_download, get_token
from huggingface_hub.utils import RepositoryNotFoundError

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1" 
SAVE_ROOT = "./train_data"
STATUS_FILE = os.path.join(SAVE_ROOT, ".download_status.json")
MAX_PARALLEL_DATASETS = 3
DOWNLOAD_RETRIES = 3

DATASET_CATEGORIES = {
    "generation": ["InternRobotics/OmniWorld", "KlingTeam/GameFactory-Dataset"],
    "reasoning": ["ai4ce/CityWalker"],
    "representation": ["annadeichler/KTH-ARIA-referential"],
    "gated": ["AgiBot/AgiBot-Demo"]
}

class StatusManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"completed": {}, "failed": {}, "verified": {}}

    def save(self, repo_id, status, error=None, info=None):
        if status == "success":
            self.data["completed"][repo_id] = {
                "time": time.time(),
                "info": info,
                "verified": False
            }
            self.data["failed"].pop(repo_id, None)
        else:
            self.data["failed"][repo_id] = {
                "error": error,
                "time": time.time(),
                "retry_count": self.data["failed"].get(repo_id, {}).get("retry_count", 0) + 1
            }
        
        with open(self.file_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def mark_verified(self, repo_id):
        if repo_id in self.data["completed"]:
            self.data["completed"][repo_id]["verified"] = True
            with open(self.file_path, 'w') as f:
                json.dump(self.data, f, indent=2)

def check_disk_space(required_bytes, path="."):
    """检查磁盘空间（预留20%缓冲）"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_percent = (free / total) * 100
        print(f"💾 磁盘可用: {free_percent:.1f}% ({free // (1024**3)}GB)")
        return free > required_bytes * 1.2
    except:
        return True

def compute_dir_checksum(directory, sample_size=100):
    """计算目录校验和（采样验证）"""
    try:
        files = list(Path(directory).rglob("*"))
        if not files:
            return None
        
        # 采样验证（大目录不全部扫描）
        sample = files[:min(sample_size, len(files))]
        hasher = hashlib.md5()
        
        for f in sorted(sample):
            if f.is_file():
                with open(f, 'rb') as file:
                    hasher.update(file.read(1024))  # 只读前1KB
        
        return hasher.hexdigest()
    except Exception as e:
        print(f"⚠️ 校验计算失败: {e}")
        return None

def verify_dataset(local_dir, repo_id, api, token):
    """多层验证：文件完整性、README、结构"""
    try:
        # 1. 检查目录非空
        if not any(Path(local_dir).iterdir()):
            return False, "目录为空"
        
        # 2. 检查是否存在标志性文件
        has_readme = any(Path(local_dir).glob("README.md"))
        file_count = len(list(Path(local_dir).rglob("*")))
        
        if file_count == 0:
            return False, "未检测到文件"
        
        # 3. 远程对比（检查是否部分下载）
        try:
            repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
            remote_count = len([s for s in repo_info.siblings if s.size])
            
            # 如果本地文件数 < 远程的50%，可能下载不完整
            if file_count < remote_count * 0.5:
                return False, f"文件数不匹配: 本地{file_count}个 < 远程{remote_count}个"
        except:
            pass
        
        # 4. 校验和
        checksum = compute_dir_checksum(local_dir)
        
        return True, {
            "file_count": file_count,
            "has_readme": has_readme,
            "checksum": checksum
        }
    except Exception as e:
        return False, str(e)

def download_dataset(repo_id, category, status_mgr, is_gated=False):
    if repo_id in status_mgr.data["completed"]:
        print(f"⏭️  跳过已完成: {repo_id}")
        return True

    local_dir = os.path.join(SAVE_ROOT, category, repo_id.replace("/", "_"))
    os.makedirs(local_dir, exist_ok=True)
    
    api = HfApi()
    token = get_token()

    # 检查gated数据集的访问权限
    if is_gated:
        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
            print(f"✓ 已验证gated数据集访问权限: {repo_id}")
        except RepositoryNotFoundError:
            error_msg = f"无访问权限或未接受条款。请访问 https://huggingface.co/datasets/{repo_id}"
            print(f"❌ {error_msg}")
            status_mgr.save(repo_id, "failed", error=error_msg)
            return False
        except Exception as e:
            print(f"⚠️ 权限检查失败: {e}")

    # 预获取信息 & 检查空间
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
        total_size = sum(s.size for s in repo_info.siblings if s.size)
        print(f"📊 数据集大小: {total_size / (1024**3):.2f}GB")
        
        if total_size > 0 and not check_disk_space(total_size, SAVE_ROOT):
            print(f"❌ 空间不足: {repo_id}")
            status_mgr.save(repo_id, "failed", error="磁盘空间不足")
            return False
    except Exception as e:
        print(f"⚠️ 无法预获取信息: {e}")
        total_size = 0

    # 执行下载 (指数退避)
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            print(f"⬇️  [{attempt}/{DOWNLOAD_RETRIES}] 正在下载: {repo_id}")
            snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=token,
                resume_download=True
            )
            
            # 验证
            verified, info = verify_dataset(local_dir, repo_id, api, token)
            if verified:
                print(f"✅ 成功: {repo_id} | 文件数: {info['file_count']}")
                status_mgr.save(repo_id, "success", info={
                    "size": total_size,
                    "file_count": info['file_count'],
                    "checksum": info['checksum']
                })
                status_mgr.mark_verified(repo_id)
                return True
            else:
                print(f"⚠️ 验证失败: {info}")
                if attempt < DOWNLOAD_RETRIES:
                    wait_time = min(2 ** attempt, 30)  # 指数退避，最多30s
                    print(f"   等待 {wait_time}s 后重试...")
                    time.sleep(wait_time)
        except Exception as e:
            print(f"⚠️ 尝试 {attempt} 失败: {e}")
            if attempt < DOWNLOAD_RETRIES:
                wait_time = min(2 ** attempt, 30)
                time.sleep(wait_time)

    status_mgr.save(repo_id, "failed", error=f"多次重试后仍失败")
    return False

def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    status_mgr = StatusManager(STATUS_FILE)
    
    try:
        import hf_transfer
        print("⚡ 已启用 hf_transfer 高速模式")
    except ImportError:
        print("💡 提示: pip install hf-transfer 可显著提速")

    start_time = time.time()
    tasks = []
    
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_DATASETS) as executor:
        for category, repos in DATASET_CATEGORIES.items():
            is_gated = category == "gated"
            for repo in repos:
                tasks.append(executor.submit(
                    download_dataset, repo, category, status_mgr, is_gated
                ))
        
        results = [t.result() for t in as_completed(tasks)]

    # 总结
    done = len(status_mgr.data["completed"])
    verified = sum(1 for v in status_mgr.data["completed"].values() if v.get("verified"))
    fail = len(status_mgr.data["failed"])
    
    print(f"\n{'='*50}")
    print(f"📊 任务完成！耗时: {int(time.time()-start_time)}s")
    print(f"✅ 成功: {done} | ✓ 已验证: {verified} | ❌ 失败: {fail}")
    print(f"📁 存储目录: {os.path.abspath(SAVE_ROOT)}")
    
    if fail > 0:
        print(f"\n⚠️ 失败列表:")
        for repo, info in status_mgr.data["failed"].items():
            print(f"   • {repo}: {info['error']}")
    
    print(f"{'='*50}")

if __name__ == "__main__":
    main()