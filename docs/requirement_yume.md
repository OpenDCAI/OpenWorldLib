# 当前仓库环境可能存在的问题

```bash
pip install torch==2.5.1 torchvision torchaudio
pip install "flash-attn==2.5.9.post1" --no-build-isolation
cd SceneFlow/
pip install -e .
```

第二步会报错，原因是缺少 psutil 这个包，补充安装之后正常。

修复方式（任选其一）：
1. 调换 `pip install -e .` 和 `flash-attn` 的安装顺序
2. 安装 `flash-attn` 前补加 `pip install psutil`

# Yume
decord
librosa