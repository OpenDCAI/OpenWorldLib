We provide a small set of test cases hosted on HuggingFace Datasets.

you can run:

```bash
git clone https://huggingface.co/datasets/YF0224/demo
```

`connectivity_graphs.pkl` 包含了每个 Matterport 场景中可通行视点（viewpoints）之间的拓扑连通关系，而非完整的场景几何结构。其中不包含 RGB、Depth 等任何可直接感知的视觉信息。

如果需要仿真数据，则需要以research名义进行申请：[Matterport3D: Learning from RGB-D Data in Indoor Environments](https://niessner.github.io/Matterport/)
