# Dataset Downloader 使用指南

用于批量下载 HuggingFace Dataset，支持断点续传、并行下载与基础完整性校验。

---

## 依赖

```bash
pip install huggingface_hub
# 可选加速
pip install hf-transfer
```

登录HuggingFace（gated 数据集必需）：

```
huggingface-cli login
```

---

## 配置

### 1. 配置数据集列表

在脚本中修改：

```python
DATASET_CATEGORIES = {
    "generation": ["org/dataset_a"],
    "reasoning": ["org/dataset_b"],
    "gated": ["org/dataset_c"]
}
```

`gated` 分类下的数据集会自动进行权限检查。

---

### 2. 下载参数（可选）

```python
MAX_PARALLEL_DATASETS = 3   # 并行下载数
DOWNLOAD_RETRIES = 3       # 失败重试次数
SAVE_ROOT = "./train_data" # 本地存储目录
```

---

## 使用

直接运行：

```bash
python download_datasets.py
```

* 已完成的数据集会自动跳过
* 失败任务支持重跑恢复
* 下载前会检查磁盘空间

---

## 下载结果

数据集按类别保存：

```text
train_data/
├── generation/
├── reasoning/
├── gated/
└── .download_status.json
```

`.download_status.json` 记录成功 / 失败 / 已验证状态。

---

## 功能简述

* 并行下载 HuggingFace Dataset
* 支持断点续传
* gated dataset 权限检查
* 磁盘空间预检测
* 下载后基础校验（文件数 + 抽样校验）
