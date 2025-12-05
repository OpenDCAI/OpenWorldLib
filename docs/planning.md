# sceneflow coarse document

## Todo

后续所有的输入都要进过pipeline的operators来输入给pipeline，operators记录所有未处理的输入

hunyuan_world_voyager需要修改输出

需要设计operators，representation，synthesis中元素具体的模版，整套设计流程具体规范需要向所有人说明

reasoning应该针对的是空间理解模型，例如：
- MindJourney: https://arxiv.org/pdf/2507.12508
- Think with 3D: https://arxiv.org/pdf/2510.18632
- https://yunpeng1998.github.io/PE-Field-HomePage/
- cambrian-S: https://arxiv.org/pdf/2511.04670

representation注意depthanything系列的工作

src/sceneflow/representations/models/utils3d need to rename as src/sceneflow/representations/models/EasternJournalist_utils3d/..

一些通用blocks需要统一，类似于matrix-game与recammaster都调用了wan

另外加入memory部分，记录过程中的不同模态中间过程

operator中除了process_interaction之外，类似于对于图片、视频的处理，应该改一个新的函数process_perception

一些通用的函数，但不包含在具体的方法类别中的，例如FlowMatchScheduler，后续统一归并到一个大类里面，src/utils中

## Student update
亦凡：Emu3.5的环境我已经测试好了，高版本的Transformers库可以正常运行（测试了4.49.0、4.50.1、4.51.0） 但是最新的4.57.1目前还存在一点问题
但是使用大于等于4.49.0的transformers时需要对代码中的函数做一点小修改：
将文件/src/emu3p5/modeling_emu3.py中类Emu3ForCausalLM的函数prepare_inputs_for_generation使用的get_max_len()改为get_max_cache_shape()
在1300行左右 

## Structure

### Note
- 区分synthesis还是representation的方式是，representation是生成固定资产（比如3D、depth、pano图），synthesis是利用模型生成的一些动态结果。

- 对于一些unified model，我们还是根据模型更倾向于什么，就不单独开一个路径承载unified model了，以bagel为例，bagel应该生成能力比较强，放在synthesis会更合适


### structure details
- sceneflow
  - operators (前后左右移动，text，相机坐标)
    - 首先夹在对应的模型有哪些操作模型，如何切换对应的操作模式
    - 记录需要进一步完成的操作
  - represetations (3D point, depth, warped image, panoramic image)
    - 需要完成对operator信号的处理，以及生成输入给rendering model的中间结果，以及记录中间结果
    - 3D scene optimization
    - 动态信号管理 （例如3D点云动态信号，具身智能以及driving的输出）
    - memory (中间存在的信号，可以是图片，应出现的交互内容) (这个或许考虑和representation放在一起)
  - synthesis (加载视频生成模型，或者inpainting模型，图片编辑模型等, or 3D rendering method)
    - visual_generation (视频生成、inpainting、图片编辑)
    - geometry_rendering (3D渲染方法)
    - audio_synthesis
  - reasoning (原knowledge)
    - semantic_planning (大模型提示该出现什么)
    - spatial_computation (3D相关运算)
    - MindJourney: https://github.com/UMass-Embodied-AGI/MindJourney
  - pipeline
    - hunyuan-world-voyager: 
    - wonderjourney: https://github.com/KovenYu/WonderJourney
    - emu3.5: https://github.com/baaivision/Emu3.5
    - wonderworld: https://github.com/KovenYu/WonderWorld
    - hunyuan-world-mirror+video generation model: https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror
    - matrix-game: https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-2
    - recammaster: https://github.com/KwaiVGI/ReCamMaster
    - evoworld: https://github.com/JiahaoPlus/EvoWorld
    - FlashWorld: https://github.com/imlixinyang/FlashWorld (3D scene generation method)
    - open-RTFM: https://www.worldlabs.ai/blog/rtfm     看看能不能训练一个实时生成的navigation model (其实就是一个实时的inpainting模型)
    - open-Marble (李飞飞推的世界模型，一种reasoning -> representation的典范，先生成布局，再生成3D效果)
    - Latticeworld: https://arxiv.org/abs/2509.05263    看看能不能调用unreal5 API实现一个大概场景生成
    - worldedit: 
    - 其他的类似FantasyWorld如果没开源，可以考虑根据已有代码复现，open-RTFM以及
    - sam-3D: https://github.com/facebookresearch/sam-3d-objects
- data
  - demo
  - load existing dataset
    - game factory: https://huggingface.co/datasets/KwaiVGI/GameFactory-Dataset
- training_example (有了这么多pipeline，你得有可以训练的环境，别人才更像star你)
  - RL (RL is important, the data is hard to get)
    - 具体策略可以是，利用视频生成模型生成一段视频，符合运动结果的reward为1
  - data_construction
