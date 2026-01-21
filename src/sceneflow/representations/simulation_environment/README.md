# simulation environment

## [ai2thor](https://ai2thor.allenai.org/)

这个是unity渲染的3d场景房间，如果要测试只能在线测试，也跟habitat-sim一样，只有单纯3d场景，没有轨迹之类的其他数据，但是可以前后左右移动并与物体交互。[huggingface](https://huggingface.co/datasets/YF0224/demo)有我录制的一个视频。

## [VLN-CE](https://jacobkrantz.github.io/vlnce/)

这个数据集也是基于matterport的，其中 `connectivity_graphs.pkl` 包含了每个 Matterport 场景中可通行视点（viewpoints）之间的拓扑连通关系，而非完整的场景几何结构。其中不包含 RGB、Depth 等任何可直接感知的视觉信息。但是他的数据集包含运动轨迹。

如果需要仿真数据，则需要以research名义进行申请：[Matterport3D: Learning from RGB-D Data in Indoor Environments](https://niessner.github.io/Matterport/)

[huggingface](https://huggingface.co/datasets/YF0224/demo)里面有 `connectivity_graphs.pkl`以及json文件格式。

## [habitat-sim](https://github.com/facebookresearch/habitat-sim)

里面包含了glb跟navmesh文件，可以仿真世界，但是不像vlnce那样具备goal跟trajectory，如果需要trajectory，需要什么matterport数据集进行渲染仿真。[huggingface](https://huggingface.co/datasets/YF0224/demo)里面有可用于测试的glb跟navmesh文件。
