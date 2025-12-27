import os
from sceneflow.pipelines.hunyuan_mirror.pipeline_hunyuan_mirror import HunyuanMirrorPipeline

# 设置输入输出路径
input_path = "data/test_case/test_image_seq_case1"
output_path = "output/hunyuan_mirror_mirror"

# 获取所有图片文件
image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
image_paths = []
for ext in image_extensions:
    image_paths.extend([os.path.join(input_path , f) for f in os.listdir(input_path ) 
                        if f.lower().endswith(ext)])
if not image_paths:
    print(f"❌ 目录中没有找到图片文件: {input_path }")
    exit(1)

# 加载模型
pipeline = HunyuanMirrorPipeline.from_pretrained(
    model_path="tencent/HunyuanWorld-Mirror",
    output_path=output_path,
    device="cuda"
)

# 运行pipeline
processing_results = pipeline(
    input_paths=image_paths,
    output_path=output_path
)

# 保存结果
results = pipeline.save_results(
    results=processing_results,
    save_pointmap=True,
    save_depth=True,
    save_normal=True
)

print("3D重建完成！")
print(f"结果已保存到: {output_path}")