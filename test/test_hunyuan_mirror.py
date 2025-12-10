#!/usr/bin/env python3
"""
HunyuanMirrorPipeline 演示脚本
"""

import os
import sys
import argparse
from PIL import Image
import numpy as np

from src.sceneflow.pipelines.hunyuan_mirror.pipeline_hunyuan_mirror import HunyuanMirrorPipeline


def main():
    parser = argparse.ArgumentParser(description="HunyuanMirrorPipeline 演示脚本")
    parser.add_argument("--input_path", type=str, required=True, 
                        help="输入路径：可以是单张图片、图片目录或视频文件")
    parser.add_argument("--output_path", type=str, default="./output/hunyuan_mirror_demo",
                        help="输出结果目录")
    parser.add_argument("--model_path", type=str, default="tencent/HunyuanWorld-Mirror",
                        help="模型路径，支持HuggingFace模型名称或本地路径")
    parser.add_argument("--local_model_path", type=str, default=None,
                        help="本地模型路径（优先级高于model_path）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备选择：cuda或cpu")
    parser.add_argument("--fps", type=int, default=1,
                        help="视频提取帧率")
    parser.add_argument("--target_size", type=int, default=518,
                        help="图像目标大小")
    
    # 过滤参数
    parser.add_argument("--confidence_percentile", type=float, default=10.0,
                        help="置信度过滤百分位")
    parser.add_argument("--edge_normal_threshold", type=float, default=5.0,
                        help="法线边缘阈值（度）")
    parser.add_argument("--edge_depth_threshold", type=float, default=0.03,
                        help="深度边缘阈值")
    parser.add_argument("--apply_confidence_mask", action="store_true", default=True,
                        help="应用置信度掩码")
    parser.add_argument("--apply_edge_mask", action="store_true", default=True,
                        help="应用边缘掩码")
    parser.add_argument("--apply_sky_mask", action="store_true", default=False,
                        help="应用天空掩码")
    
    # 保存选项
    parser.add_argument("--save_pointmap", action="store_true", default=True,
                        help="保存点云")
    parser.add_argument("--save_depth", action="store_true", default=True,
                        help="保存深度图")
    parser.add_argument("--save_normal", action="store_true", default=True,
                        help="保存法线图")
    parser.add_argument("--save_gs", action="store_true", default=True,
                        help="保存高斯分布")
    parser.add_argument("--save_rendered", action="store_true", default=True,
                        help="保存渲染视频")
    parser.add_argument("--save_colmap", action="store_true", default=True,
                        help="保存COLMAP重建")
    
    args = parser.parse_args()
    
    print("🚀 启动HunyuanMirrorPipeline演示")
    print(f"📥 输入路径: {args.input_path}")
    print(f"📤 输出路径: {args.output_path}")
    print(f"🤖 模型路径: {args.model_path}")
    print(f"💻 设备: {args.device}")
    
    # 1. 初始化pipeline
    try:
        print("\n📦 加载模型...")
        pipeline = HunyuanMirrorPipeline.from_pretrained(
            model_path=args.model_path,
            local_model_path=args.local_model_path,
            output_path=args.output_path,
            device=args.device
        )
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 2. 准备输入数据
    print("\n📸 准备输入数据...")
    input_data = None
    image_paths = None
    
    assert os.path.isdir(args.input_path)
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend([os.path.join(args.input_path, f) for f in os.listdir(args.input_path) 
                            if f.lower().endswith(ext)])
    
    if not image_paths:
        print(f"❌ 目录中没有找到图片文件: {args.input_path}")
        return
    
    image_paths.sort()
    print(f"✅ 找到 {len(image_paths)} 张图片")
    
    # 3. 运行pipeline
    print("\n🔄 开始3D重建...")
    try:
        processing_results = pipeline(
            input_paths=image_paths,
            output_path=args.output_path,
            confidence_percentile=args.confidence_percentile,
            edge_normal_threshold=args.edge_normal_threshold,
            edge_depth_threshold=args.edge_depth_threshold,
            apply_confidence_mask=args.apply_confidence_mask,
            apply_edge_mask=args.apply_edge_mask,
            apply_sky_mask=args.apply_sky_mask
        )
        
        results = pipeline.save_results(
            results=processing_results,
            save_pointmap=args.save_pointmap,
            save_depth=args.save_depth,
            save_normal=args.save_normal,
            save_gs=args.save_gs,
            save_rendered=args.save_rendered,
            save_colmap=args.save_colmap
        )
        
        print("\n✅ 3D重建完成！")
        print("\n📊 生成的结果:")
        for key, value in results.items():
            print(f"  - {key}: {value}")
            
        print(f"\n🎉 所有结果已保存到: {args.output_path}")
        
    except Exception as e:
        print(f"❌ 3D重建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 显示结果预览
    print("\n👀 结果预览:")
    if args.save_depth:
        depth_dir = os.path.join(args.output_path, "depth")
        if os.path.exists(depth_dir):
            depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.png')]
            if depth_files:
                print(f"  深度图: {os.path.join(depth_dir, depth_files[0])}")
    
    if args.save_pointmap:
        pointmap_path = os.path.join(args.output_path, "pts_from_pointmap.ply")
        if os.path.exists(pointmap_path):
            print(f"  点云: {pointmap_path}")
    
    if args.save_gs:
        gs_path = os.path.join(args.output_path, "gaussians.ply")
        if os.path.exists(gs_path):
            print(f"  高斯分布: {gs_path}")
    
    print("\n📚 可以使用以下工具查看结果:")
    print("  - 点云/Ply文件: MeshLab, CloudCompare")
    print("  - 深度图: 任何图片查看器或3D软件")
    print("  - 渲染视频: 任何视频播放器")
    print("  - COLMAP结果: COLMAP软件")


if __name__ == "__main__":
    main()