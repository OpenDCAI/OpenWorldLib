wonder_world_config = {
    # 基础配置
    "runs_dir": "output/real_campus_2",
    "example_name": "real_campus_2",
    "seed": 1,
    "device": "cuda",
    "debug": False,
    
    # GPT和生成配置
    "use_gpt": False,
    "api_key": "Your OpenAI api_key",
    
    # 深度模型配置
    "depth_model": "marigold",  # choice: [MiDaS, ZoeDepth, MariGold]
    "depth_conditioning": True,
    
    # 相机参数
    "camera_speed": 0.001,
    "init_focal_length": 500,  # 使用后面定义的值
    "rotation_range": 0.111,  # 使用后面定义的值，原始: 0.37
    "camera_speed_multiplier_rotation": 0.3,
    
    # 深度参数
    "fg_depth_range": 0.015,
    "depth_shift": 0.001,
    "sky_hard_depth": 0.02,
    "sky_erode_kernel_size": 0,
    "ground_erode_kernel_size": 3,
    "dilate_mask_decoder_ft": 3,
    
    # 天空和图层生成
    "gen_sky_image": False,
    "gen_sky": False,
    "gen_layer": True,
    
    # 加载配置
    "load_gen": False,
    
    # 帧数配置
    "num_frames": 1,
    "frames": 5,
    "num_scenes": 16,
    "save_fps": 30,
    
    # Inpainting参数
    "negative_inpainting_prompt": "collage, text, writings, signs, text, white border, photograph border, artifacts, blur, blurry, foggy, fog, person, bad quality, distortions, distorted image, watermark, signature, fisheye look",
    "inpainting_resolution_interp": 512,
    "inpainting_resolution_gen": 512,
    
    # Decoder微调参数
    "finetune_depth_decoder": False,
    "decoder_learning_rate": 0.0001,
    "num_finetune_decoder_steps": 100,
    "num_finetune_decoder_steps_interp": 30,
    "preservation_weight": 10,
    
    # 深度微调参数
    "depth_model_learning_rate": 1e-6,
    
    # 跳过和重新生成配置
    "skip_interp": False,
    "skip_gen": False,
    "kf2_upsample_coef": 4,
    "regenerate_times": 3,
    
    # 其他配置
    "use_compile": False,
    "use_free_lunch": False,
    "rotation_path": [2, 2, 2, 2, 2, 2, 2, 2, -2, -2, -2, -2, -2, -2, -2, -2],
}
