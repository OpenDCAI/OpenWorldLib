import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional

# VLA evaluation imports
try:
    os.environ['MUJOCO_GL'] = 'osmesa'
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False


def reference_func(
    pipe,
    pipe_infer,
    input_data_info: Dict[str, Any],
    output_key: str = "generated_actions"
) -> Dict[str, Any]:
    """
    根据 input_data_info（由 BenchmarkLoader 组装的单条测例），
    驱动 SpiritV1p5Pipeline 生成机器人动作序列。

    Args:
        pipe:            已初始化的 SpiritV1p5Pipeline 实例。
        pipe_infer:      VLA pipeline 推理函数。
        input_data_info: 单条测例字典，至少包含
                         - main_view:            主视角图片的绝对路径（str）
                         - wrist_view:           手腕视角图片的绝对路径（str）
                         - raw_state:            机器人初始状态列表
                         - task:                 任务描述字符串
                         - robot_type:           机器人类型（如 "Franka"）
                         可选：
                         - output_path:          若提供，则将动作序列保存到该路径（JSON格式）
        output_key:      输出字典中存放生成动作的键名。

    Returns:
        {output_key: 保存后的动作文件路径}（当 input_data_info 含 output_path 时）
        或 {output_key: None, "error": 错误信息}（如果执行失败）
    """
    
    # 1. 加载输入图像
    main_view_path = input_data_info.get("main_view")
    wrist_view_path = input_data_info.get("wrist_view")
    
    if not main_view_path or not wrist_view_path:
        return {
            output_key: None,
            "error": "Missing main_view or wrist_view path"
        }
    
    try:
        images = {
            "cam_high": Image.open(main_view_path).convert("RGB"),
            "cam_left_wrist": Image.open(wrist_view_path).convert("RGB"),
        }
    except Exception as e:
        return {
            output_key: None,
            "error": f"Failed to load images: {str(e)}"
        }
    
    # 2. 获取其他输入参数
    raw_state = input_data_info.get("raw_state")
    task = input_data_info.get("task")
    robot_type = input_data_info.get("robot_type", "Franka")
    
    if raw_state is None or task is None:
        return {
            output_key: None,
            "error": "Missing raw_state or task"
        }
    
    # 兼容 raw_state 为 JSON 字符串的情况
    if isinstance(raw_state, str):
        try:
            raw_state = json.loads(raw_state)
        except json.JSONDecodeError:
            return {
                output_key: None,
                "error": "Invalid raw_state format"
            }
    
    # 3. 使用 Pipeline 生成动作序列
    try:
        action_sequence = pipe_infer(
            pipe=pipe,
            images=images,
            raw_state=raw_state,
            task=task,
            robot_type=robot_type,
            return_all_steps=True,
        )
        
        # 转换为可序列化的格式（numpy array -> list）
        if hasattr(action_sequence, 'tolist'):
            action_sequence = action_sequence.tolist()
        elif isinstance(action_sequence, list):
            action_sequence = [
                action.tolist() if hasattr(action, 'tolist') else action
                for action in action_sequence
            ]
            
    except Exception as e:
        return {
            output_key: None,
            "error": f"Action generation failed: {str(e)}"
        }
    
    # 4. 保存动作序列
    output_path = input_data_info.get("output_path", None)
    if output_path is None:
        return {
            output_key: None,
            "error": "output_path is required for VLA evaluation"
        }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w') as f:
            json.dump({
                "actions": action_sequence,
                "task": task,
                "robot_type": robot_type,
                "num_steps": len(action_sequence)
            }, f, indent=2)
        
        return {output_key: str(output_path)}
    except Exception as e:
        return {
            output_key: None,
            "error": f"Failed to save actions: {str(e)}"
        }


def eval_func(
    input_data_info: Dict[str, Any],
    eval_pipeline: None,
    eval_pipeline_infer: None,
) -> Dict[str, Any]:
    """
    在 LIBERO 环境中执行生成的动作序列，检查任务是否成功完成。
    
    Args:
        input_data_info: 单条测例字典，包含：
            - generated_actions_path: 生成的动作序列文件路径（从 reference_results 传入）
            - task: 任务描述
            - benchmark_name: LIBERO benchmark 名称（如 "libero_10"）
            - norm_stats: 归一化统计数据（可选）
            - bddl_file_name: BDDL 文件路径（可选）
            - init_state: 初始状态索引（默认 0）
        eval_pipeline: 此处不使用，保留以兼容接口
        eval_pipeline_infer: 此处不使用，保留以兼容接口
    
    Returns:
        包含评估结果的字典：
        {
            'sample_id': str,
            'generated_actions_path': str,
            'success': bool,
            'success_step': int or None,  # 任务成功的步数（如果成功）
            'total_steps': int
        }
    """
    # 检查 LIBERO 是否可用
    if not LIBERO_AVAILABLE:
        return {
            'sample_id': input_data_info.get('id', 'unknown'),
            'error': 'LIBERO library not available. Please install: pip install libero'
        }
    
    generated_actions_path = input_data_info.get("generated_actions_path")
    if not generated_actions_path:
        raise ValueError("generated_actions_path not found in input_data_info")
    
    # 验证文件存在
    if not Path(generated_actions_path).exists():
        raise FileNotFoundError(f"Actions file not found: {generated_actions_path}")
    
    # 读取动作序列
    try:
        with open(generated_actions_path, 'r') as f:
            actions_data = json.load(f)
            action_sequence = actions_data["actions"]
    except Exception as e:
        return {
            'sample_id': input_data_info.get('id', 'unknown'),
            'generated_actions_path': generated_actions_path,
            'error': f"Failed to load actions: {str(e)}"
        }
    
    # 获取环境配置
    task_description = input_data_info.get("task")
    benchmark_name = input_data_info.get("benchmark_name", "libero_10")
    norm_stats = input_data_info.get("norm_stats")
    bddl_file_name = input_data_info.get("bddl_file_name")
    init_state_idx = input_data_info.get("init_state", 0)
    
    # 执行环境评估
    try:
        result = _execute_in_libero_env(
            action_sequence=action_sequence,
            task_description=task_description,
            benchmark_name=benchmark_name,
            norm_stats=norm_stats,
            bddl_file_name=bddl_file_name,
            init_state_idx=init_state_idx,
        )
        
        return {
            'sample_id': input_data_info.get('id', 'unknown'),
            'generated_actions_path': generated_actions_path,
            'success': result['success'],
            'success_step': result.get('success_step'),
            'total_steps': result['total_steps'],
        }
        
    except Exception as e:
        return {
            'sample_id': input_data_info.get('id', 'unknown'),
            'generated_actions_path': generated_actions_path,
            'error': f"Environment execution failed: {str(e)}"
        }


def _execute_in_libero_env(
    action_sequence: list,
    task_description: str,
    benchmark_name: str,
    norm_stats: Optional[Dict] = None,
    bddl_file_name: Optional[str] = None,
    init_state_idx: int = 0,
) -> Dict[str, Any]:
    """
    在 LIBERO 环境中执行动作序列（不录制视频）。
    
    Returns:
        Dict: {
            'success': bool,
            'success_step': int or None,
            'total_steps': int
        }
    """
    # 1. 获取任务信息
    bm = benchmark.get_benchmark(benchmark_name)()
    
    # 查找任务
    task_id = None
    task_obj = None
    for i in range(bm.get_num_tasks()):
        task = bm.get_task(i)
        if task_description.lower() in task.language.lower():
            task_id = i
            task_obj = task
            break
    
    if task_id is None:
        raise ValueError(f"Task '{task_description}' not found in {benchmark_name}")
    
    # 2. 初始化环境
    if bddl_file_name:
        env_bddl_path = bddl_file_name
    else:
        # 使用相对路径从 data 目录构建 bddl 文件路径
        # 获取项目根目录（从当前文件向上2级到项目根目录）
        project_root = Path(__file__).resolve().parent.parent.parent
        bddl_base = project_root / "data" / "benchmarks" / "vla" / "vla_libero_test"
        env_bddl_path = str(bddl_base / task_obj.problem_folder / task_obj.bddl_file)
    
    env_args = {
        "bddl_file_name": env_bddl_path,
        "camera_heights": 256,
        "camera_widths": 256,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "render_gpu_device_id": 0,
    }
    
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    
    # 3. 设置初始状态
    env.reset()
    init_states = bm.get_task_init_states(task_id)
    env.set_init_state(init_states[init_state_idx])
    obs = env.reset()
    
    # 4. 执行动作序列（不录制视频）
    success = False
    success_step = None
    
    for i, raw_action in enumerate(action_sequence):
        # 处理动作（如果需要反归一化）
        if norm_stats is not None:
            env_action = _process_action(raw_action, norm_stats)
        else:
            # 直接使用前 7 维作为环境动作
            env_action = np.array(raw_action[:7])
        
        # 执行动作
        obs, reward, done, info = env.step(env_action)
        
        # 检查任务是否成功
        if env.check_success() and not success:
            success = True
            success_step = i + 1
            # 可以选择在此处 break，或继续执行完所有动作
    
    # 5. 清理
    env.close()
    
    return {
        'success': success,
        'success_step': success_step,
        'total_steps': len(action_sequence)
    }


def _process_action(raw_action: list, stats: Dict) -> np.ndarray:
    """
    处理模型输出的动作：填充、反归一化、截取。
    
    Args:
        raw_action: 模型输出的动作（通常是 8 维）
        stats: 归一化统计数据，包含 "actions" -> {"mean": [...], "std": [...]}
    
    Returns:
        环境所需的动作（7 维）
    """
    # 1. 准备统计量（确保长度为 32）
    mean = np.array(stats["actions"]["mean"])
    std = np.array(stats["actions"]["std"])
    
    target_dim = 32
    if len(mean) < target_dim:
        mean = np.pad(mean, (0, target_dim - len(mean)), 'constant')
    if len(std) < target_dim:
        std = np.pad(std, (0, target_dim - len(std)), 'constant')
    
    # 2. 填充输入到 32 维
    action_in = np.array(raw_action)
    action_full = np.zeros(target_dim)
    current_dim = min(len(action_in), target_dim)
    action_full[:current_dim] = action_in[:current_dim]
    
    # 3. 反归一化
    unnormalized_full = (action_full * std) + mean
    
    # 4. 截取前 7 维（环境所需）
    env_action = unnormalized_full[:7]
    
    return env_action

