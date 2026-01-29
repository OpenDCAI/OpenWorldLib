

<div align="center" markdown="1">

### A Unified Framework Collecting Advanced World Models

</div>

### Table of Contents

- [A Unified Framework Collecting Advanced World Models](#a-unified-framework-collecting-advanced-world-models)
- [Table of Contents](#table-of-contents)
- [Features](#features)
- [Quick Start](#quick-start)
  - [Install](#install)
- [Structure](#structure)
- [Planning](#planning)
- [For Developers](#for-developers)
- [Citation](#citation)


### Features


### Quick Start
#### Install
首先安装torch以及flash_attn
```bash
pip install torch==2.5.1 torchvision torchaudio
pip install "flash-attn==2.5.9.post1" --no-build-isolation
```
接着安装不同版本的sceneflow，进入`./SceneFlow`，下面分别是高版本transformers安装`[transformers_high]`以及低版本transformers安装`[transformers_low]`，推荐安装高版本transformers:
```bash
pip install -e ".[transformers_high]"
```
下面是低版本transformers环境安装
```bash
pip install -e ".[transformers_low]"
```

### Structure


### Planning


### For Developers


### Citation
