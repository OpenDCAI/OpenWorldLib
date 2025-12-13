# HuggingFace 数据集批量下载工具

> 🚀 专为 World Model / 计算机视觉 / 多模态大模型设计的大规模数据集下载解决方案

## 📋 目录

- [功能特性](#功能特性)
- [安装配置](#安装配置)
- [快速开始](#快速开始)
- [核心功能详解](#核心功能详解)
- [命令行参数](#命令行参数)
- [配置选项](#配置选项)
- [使用场景](#使用场景)
- [常见问题](#常见问题)
- [最佳实践](#最佳实践)

---

## 功能特性

### ✨ 核心能力

- **🔄 断点续传**: 支持中断后继续下载，避免重复下载
- **⚡ 并行下载**: 最多同时下载 N 个数据集，充分利用带宽
- **🔐 智能认证**: 自动检测公共/私有数据集，按需使用 token
- **💾 磁盘预检**: 下载前检查空间，避免下载到一半失败
- **🎯 文件过滤**: 支持白名单/黑名单，只下载需要的文件类型
- **🔍 数据校验**: 多层次完整性验证，确保数据质量
- **📊 详细报告**: 下载和校验结果全程记录，可追溯

### 🎯 特别优化

针对 **大规模视频数据集**（几十 GB - 几 TB）的场景优化：

- 启用 `hf_transfer` 高速传输协议
- 智能重试机制（网络波动自动恢复）
- 元数据持久化（避免重复下载）
- 分类管理（按用途组织数据集）

---

## 安装配置

### 环境要求

- Python 3.8+
- 稳定的网络连接
- 足够的磁盘空间

### 依赖安装

```bash
# 基础依赖
pip install huggingface_hub

# 高速传输（强烈推荐！可提升 3-5 倍下载速度）
pip install hf-transfer

# 如果需要处理视频元数据
pip install opencv-python  # 可选
```

### HuggingFace 登录

```bash
# 首次使用需要登录（用于下载私有/门禁数据集）
huggingface-cli login

# 输入你的 Access Token（从 https://huggingface.co/settings/tokens 获取）
```

---

## 快速开始

### 1️⃣ 配置数据集列表

编辑脚本中的 `DATASET_CATEGORIES`：

```python
DATASET_CATEGORIES = {
    "generation": [
        "InternRobotics/OmniWorld",
        "KlingTeam/GameFactory-Dataset",
    ],
    "reasoning": [
        "ai4ce/CityWalker",
    ],
    "representation": [
        "annadeichler/KTH-ARIA-referential",
    ],
    "gated": [
        "AgiBot/AgiBot-Demo",  # 需要申请权限
    ]
}
```

### 2️⃣ 基础使用

```bash
# 最简单的用法：下载所有数据集
python download_datasets.py

# 并行下载（推荐，速度快 3-5 倍）
python download_datasets.py --parallel

# 只下载特定分类
python download_datasets.py --category generation
```

### 3️⃣ 查看结果

```bash
train_data/
├── generation/
│   ├── InternRobotics_OmniWorld/
│   │   ├── video_001.mp4
│   │   ├── video_002.mp4
│   │   └── metadata.json
│   └── KlingTeam_GameFactory-Dataset/
├── reasoning/
│   └── ai4ce_CityWalker/
└── .download_metadata.json  # 下载进度记录
└── .validation_report.json  # 校验结果记录
```

---

## 核心功能详解

### 🔄 断点续传

**工作原理**：
- 每个数据集下载完成后记录状态
- 脚本中断后重新运行，自动跳过已完成的数据集
- 单个数据集内部也支持断点续传（HuggingFace Hub 原生支持）

**元数据文件**：`.download_metadata.json`

```json
{
  "completed": ["InternRobotics/OmniWorld", "ai4ce/CityWalker"],
  "failed": [
    {
      "repo": "some/failed-dataset",
      "error": "Network timeout",
      "time": 1701234567
    }
  ],
  "in_progress": {}
}
```

**使用场景**：
```bash
# 第一次运行，下载了 3 个数据集后网络中断
python download_datasets.py --parallel

# 重新运行，自动跳过已完成的，继续下载剩余的
python download_datasets.py --parallel
```

### ⚡ 并行下载

**配置参数**：

```python
MAX_PARALLEL_DATASETS = 3  # 同时下载的数据集数量
```

**命令行使用**：

```bash
# 使用默认并发数（3）
python download_datasets.py --parallel

# 自定义并发数
python download_datasets.py --parallel --max-workers 5

# 网络带宽充足时可以增加
python download_datasets.py --parallel --max-workers 10
```

**注意事项**：
- 并发数不是越大越好，建议根据带宽调整
- 家庭网络：2-3 个并发
- 企业/实验室网络：5-10 个并发
- 单个数据集内部的文件下载已经是并行的

### 🎯 文件过滤

#### 白名单模式（只下载指定类型）

```python
# 只下载视频和元数据文件
ALLOWED_EXTENSIONS = ['.mp4', '.avi', '.webm', '.mkv', '.json']
```

#### 黑名单模式（忽略文档文件）

```python
# 忽略文档和配置文件
IGNORE_PATTERNS = ["*.md", "*.txt", "*.rst", ".gitattributes", "README*"]
```

#### 实际效果

**不使用过滤**：
```
下载数据集: 50 GB (包含视频、图片、文档、代码等)
```

**使用过滤（只要视频和JSON）**：
```
下载数据集: 35 GB (只有视频和元数据)
节省空间: 15 GB (30%)
节省时间: 显著减少
```

### 🔍 数据校验

#### 校验级别

| 级别 | 检查内容 | 耗时 | 推荐场景 |
|------|---------|------|---------|
| **Level 1** | 文件数量对比 | 秒级 | 常规使用 ✅ |
| **Level 2** | 文件类型统计 | 秒级 | 常规使用 ✅ |
| **Level 3** | 关键文件检测 | 秒级 | 常规使用 ✅ |
| **Level 4** | 哈希值校验 | 分钟-小时级 | 高安全需求 |

#### 校验配置

```python
# 全局配置
ENABLE_VALIDATION = True          # 是否启用校验
ENABLE_HASH_CHECK = False         # 是否启用哈希校验（耗时）
```

#### 校验报告示例

```
🔍 开始校验数据集: InternRobotics/OmniWorld
📋 获取远程文件列表...
   远程文件数: 3046
📁 统计本地文件...
   本地文件数: 3046
   本地总大小: 45.23 GB
   文件类型分布:
      .mp4: 1523 个
      .json: 1523 个
   ✅ 文件数量校验通过
✅ 数据集校验通过: InternRobotics/OmniWorld
```

#### 校验结果文件

`.validation_report.json`:

```json
{
  "validated": {
    "InternRobotics/OmniWorld": {
      "local_file_count": 3046,
      "remote_file_count": 3046,
      "local_total_size": 48566123456,
      "file_types": {
        ".mp4": 1523,
        ".json": 1523
      },
      "passed": true,
      "timestamp": 1701234567.89
    }
  },
  "failed": {}
}
```

---

## 命令行参数

### 完整参数列表

```bash
python download_datasets.py [OPTIONS]
```

| 参数 | 说明 | 示例 |
|------|------|------|
| `--parallel` | 启用并行下载 | `--parallel` |
| `--max-workers N` | 设置并发数（默认3） | `--max-workers 5` |
| `--category NAME` | 只下载指定分类 | `--category generation` |
| `--force` | 强制重新下载已完成的数据集 | `--force` |
| `--no-validation` | 禁用下载后校验 | `--no-validation` |
| `--enable-hash-check` | 启用哈希校验（耗时） | `--enable-hash-check` |
| `--validate-only REPO` | 只校验指定数据集 | `--validate-only ai4ce/CityWalker` |

### 使用示例

```bash
# 示例 1: 标准下载（串行 + 校验）
python download_datasets.py

# 示例 2: 快速下载（并行 + 无校验）
python download_datasets.py --parallel --no-validation

# 示例 3: 高安全性下载（并行 + 哈希校验）
python download_datasets.py --parallel --enable-hash-check

# 示例 4: 只下载生成类数据集
python download_datasets.py --parallel --category generation

# 示例 5: 强制重新下载所有数据
python download_datasets.py --force

# 示例 6: 事后校验某个数据集
python download_datasets.py --validate-only "InternRobotics/OmniWorld"

# 示例 7: 高并发下载（适合高带宽）
python download_datasets.py --parallel --max-workers 10
```

---

## 配置选项

### 路径配置

```python
SAVE_ROOT = "./train_data"              # 数据集保存根目录
CACHE_ROOT = "./hf_cache"               # HuggingFace 缓存目录
METADATA_FILE = ".download_metadata.json"
VALIDATION_REPORT_FILE = ".validation_report.json"
```

### 下载配置

```python
DOWNLOAD_RETRIES = 3                    # 失败重试次数
RETRY_DELAY_SEC = 10                    # 重试延迟（秒）
MAX_PARALLEL_DATASETS = 3               # 并行下载数量
RESUME_DOWNLOAD = True                  # 断点续传
```

### 文件过滤配置

```python
# 只下载特定文件类型（None = 下载所有）
ALLOWED_EXTENSIONS = None
# 示例：只下载视频和元数据
# ALLOWED_EXTENSIONS = ['.mp4', '.avi', '.webm', '.mkv', '.json']

# 忽略的文件模式
IGNORE_PATTERNS = ["*.md", "*.txt", ".gitattributes"]
```

### 校验配置

```python
ENABLE_VALIDATION = True                # 启用下载后校验
ENABLE_HASH_CHECK = False               # 启用哈希校验（耗时）
```

---

## 使用场景

### 🎬 场景 1: 下载 World Model 训练数据

**需求**：
- 下载 10+ 个大型视频数据集
- 每个数据集 20-100 GB
- 只需要视频和标注文件

**配置**：

```python
ALLOWED_EXTENSIONS = ['.mp4', '.avi', '.webm', '.json', '.csv']
MAX_PARALLEL_DATASETS = 3
ENABLE_VALIDATION = True
ENABLE_HASH_CHECK = False  # 视频文件太大，不做哈希校验
```

**命令**：

```bash
python download_datasets.py --parallel --category generation
```

### 🤖 场景 2: 下载机器人数据集（需要权限）

**需求**：
- 部分数据集需要申请访问权限
- 需要使用 HuggingFace token

**步骤**：

1. 先登录 HuggingFace：
```bash
huggingface-cli login
```

2. 在 HuggingFace 网站上申请数据集访问权限

3. 运行下载：
```bash
python download_datasets.py --category gated
```

### 🔬 场景 3: 学术研究（需要完整性验证）

**需求**：
- 确保数据集 100% 完整
- 用于可复现的学术研究

**配置**：

```python
ENABLE_VALIDATION = True
ENABLE_HASH_CHECK = True  # 启用最严格的校验
```

**命令**：

```bash
python download_datasets.py --enable-hash-check
```

### 💾 场景 4: 断点续传（网络不稳定）

**情况**：下载过程中网络中断或手动停止

**解决方案**：

```bash
# 第一次运行，下载部分数据后中断
python download_datasets.py --parallel

# 重新运行，自动继续
python download_datasets.py --parallel
# ✅ 会自动跳过已完成的数据集
# ✅ 部分下载的数据集会从断点继续
```

### 🔍 场景 5: 事后校验已下载的数据

**情况**：
- 数据已下载，但怀疑不完整
- 需要验证数据质量

**命令**：

```bash
# 校验单个数据集
python download_datasets.py --validate-only "InternRobotics/OmniWorld"

# 校验所有数据集（不重新下载）
# 方法：临时修改代码，在主循环中只调用 validate_dataset
```

---

## 常见问题

### Q1: 下载速度很慢怎么办？

**解决方案**：

1. 安装高速传输库：
```bash
pip install hf-transfer
```

2. 增加并发数：
```bash
python download_datasets.py --parallel --max-workers 5
```

3. 检查网络：
```bash
# 测试到 HuggingFace 的连接速度
curl -o /dev/null https://huggingface.co/datasets/mnist/resolve/main/mnist.tar.gz
```

### Q2: 提示 "HuggingFace token not found"

**原因**：尝试下载需要权限的数据集，但未登录

**解决方案**：

```bash
# 登录 HuggingFace
huggingface-cli login

# 输入 token（从 https://huggingface.co/settings/tokens 获取）
```

### Q3: 磁盘空间不足

**解决方案**：

1. 修改保存路径到更大的磁盘：
```python
SAVE_ROOT = "/mnt/large_disk/train_data"
```

2. 启用文件过滤，只下载必要文件：
```python
ALLOWED_EXTENSIONS = ['.mp4', '.json']  # 只要视频和元数据
```

3. 分批下载：
```bash
# 先下载第一个分类
python download_datasets.py --category generation

# 清理后再下载第二个分类
python download_datasets.py --category reasoning
```

### Q4: 某个数据集一直下载失败

**可能原因**：
- 网络问题
- 数据集被删除或移动
- 权限问题

**解决方案**：

1. 查看错误日志：
```bash
python download_datasets.py 2>&1 | tee download.log
```

2. 单独测试这个数据集：
```bash
# 临时修改 DATASET_CATEGORIES，只保留这一个数据集
python download_datasets.py
```

3. 手动验证数据集是否存在：
访问 `https://huggingface.co/datasets/[repo_id]`

### Q5: 校验失败怎么办？

**情况 1**：文件数量不匹配

- 可能是因为启用了 `ALLOWED_EXTENSIONS`，这是正常的
- 检查是否有文件下载失败

**情况 2**：关键文件缺失

- 重新下载该数据集：
```bash
python download_datasets.py --force --validate-only "[repo_id]"
```

### Q6: 如何查看下载进度？

**方法 1**：查看终端输出

```
⏳ 正在处理数据集: InternRobotics/OmniWorld
📊 数据集信息: 3046 个文件, 总大小: 45.23 GB
⬇️  开始下载 (尝试 1/3)...
```

**方法 2**：查看元数据文件

```bash
# 实时查看已完成的数据集
cat train_data/.download_metadata.json | jq '.completed'
```

**方法 3**：使用 `watch` 命令监控

```bash
# 监控目录大小变化
watch -n 5 'du -sh train_data/*'
```

---

## 最佳实践

### 🎯 针对 World Model 的推荐配置

```python
# 配置文件
SAVE_ROOT = "/data/world_model_datasets"  # 使用大容量磁盘
CACHE_ROOT = "/data/hf_cache"
MAX_PARALLEL_DATASETS = 3                 # 保守并发

# 只下载视频和元数据
ALLOWED_EXTENSIONS = ['.mp4', '.avi', '.webm', '.mkv', '.json', '.csv']

# 忽略文档
IGNORE_PATTERNS = ["*.md", "*.txt", "*.rst", "README*", ".git*"]

# 启用校验但不做哈希（视频太大）
ENABLE_VALIDATION = True
ENABLE_HASH_CHECK = False
```

### 📋 下载前检查清单

- [ ] 确认磁盘空间充足（预留 20% 余量）
- [ ] 已安装 `hf-transfer`（提速 3-5 倍）
- [ ] 已登录 HuggingFace（如有私有数据集）
- [ ] 网络连接稳定
- [ ] 配置了合理的文件过滤规则

### 🚀 性能优化技巧

1. **使用 SSD 存储缓存**：
```python
CACHE_ROOT = "/path/to/ssd/hf_cache"  # 缓存放 SSD
SAVE_ROOT = "/path/to/hdd/train_data"  # 数据放 HDD
```

2. **分时下载**：
```bash
# 在网络空闲时段下载（如深夜）
nohup python download_datasets.py --parallel > download.log 2>&1 &
```

3. **监控资源占用**：
```bash
# 监控网络、磁盘、内存
htop
iotop
nethogs
```

### 📊 数据管理建议

1. **定期备份元数据文件**：
```bash
cp train_data/.download_metadata.json train_data/.download_metadata.backup
cp train_data/.validation_report.json train_data/.validation_report.backup
```

2. **生成数据集清单**：
```bash
# 生成已下载数据集的列表
find train_data -type d -maxdepth 2 | sort > dataset_inventory.txt
```

3. **计算总大小**：
```bash
# 统计各分类的数据量
du -sh train_data/*/
```

### 🔒 安全建议

1. **Token 安全**：
   - 不要将 token 硬编码在脚本中
   - 使用 `huggingface-cli login` 登录
   - 定期轮换 token

2. **权限管理**：
   - 申请数据集访问权限时说明用途
   - 遵守数据集的使用协议

3. **备份策略**：
   - 重要数据集下载后及时备份
   - 使用 `rsync` 同步到备份服务器

---

## 📞 支持与反馈

### 遇到问题？

1. 检查本文档的 [常见问题](#常见问题) 部分
2. 查看 HuggingFace Hub 文档：https://huggingface.co/docs/hub
3. 检查网络连接和磁盘空间

### 日志收集

```bash
# 运行时保存完整日志
python download_datasets.py --parallel 2>&1 | tee download_$(date +%Y%m%d_%H%M%S).log
```

### 贡献改进

欢迎提交改进建议：
- 添加新的校验逻辑
- 优化性能
- 支持更多数据源

---

## 📄 许可证

本工具基于 MIT 许可证开源。数据集的使用需遵守各自的许可证和使用条款。

---

## 🎉 开始使用

```bash
# 1. 安装依赖
pip install huggingface_hub hf-transfer

# 2. 登录 HuggingFace
huggingface-cli login

# 3. 配置数据集列表
# 编辑脚本中的 DATASET_CATEGORIES

# 4. 开始下载！
python download_datasets.py --parallel

# 5. 喝杯咖啡，等待下载完成 ☕
```

**祝你的 World Model 训练顺利！🚀**