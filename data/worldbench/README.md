config里面定义了配置model所需要参数

models里面用来放模型的初始化定义

results放生成的中间产物跟最后的评估结果

core里面定义了一个注册器，用于注册所需任务

data用来放所需的dataset跟json文件

evaluators是用来评估的，调用metrics里面的class来进行计算，打算结合worldscore（测质量）跟vinoground还有MLVU（这俩都是事件顺序）

generator是用于调用模型来进行直接生成的，打算集成到下面的main里面直接输入模型跟数据集还有配套参数直接命令行生成

关于navigation还没想好怎么设计进去，因为是一个需要交互的环境
