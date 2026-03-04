from PIL import Image
import torch
import imageio
import os
from sceneflow.pipelines.hunyuan_world.pipeline_hunyuan_game_craft import HunyuanGameCraftPipeline


def get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def is_dist() -> bool:
    return get_world_size() > 1


def barrier():
    if is_dist():
        torch.distributed.barrier()


def bcast_object(obj, src: int = 0):
    if get_world_size() == 1:
        return obj
    obj_list = [obj] if get_rank() == src else [None]
    torch.distributed.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def setup_stdin_guard():
    if get_rank() != 0:
        sys.stdin = open(os.devnull, "r")


def safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return "q"


def parse_signals(s: str, allowed):
    sigs = [x.strip() for x in s.split(",") if x.strip()]
    invalid = [x for x in sigs if x not in allowed]
    return sigs, invalid


def parse_speeds(s: str, n: int):
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if len(parts) == 0:
        raise ValueError("Empty speed input")

    speeds = [float(x) for x in parts]
    if len(speeds) == 1 and n > 1:
        speeds = speeds * n

    if len(speeds) != n:
        raise ValueError(f"Speed count mismatch: got {len(speeds)}, expected {n} (or 1)")

    for v in speeds:
        if not (0.0 <= v <= 3.0):
            raise ValueError(f"Speed out of range [0,3]: {v}")

    return speeds


def make_next_cmd(turn_idx: int, allowed_interactions):
    rank = get_rank()

    if rank == 0:
        s = safe_input(
            f"\n[Turn {turn_idx}] Enter interaction(s) (or 'n'/'q' to stop): "
        ).strip().lower()

        if s in ["n", "q"]:
            cmd = {"type": "stop", "turn": turn_idx}
        else:
            signals, invalid = parse_signals(s, allowed_interactions)
            if invalid or not signals:
                cmd = {
                    "type": "invalid",
                    "turn": turn_idx,
                    "err": f"Invalid interaction(s): {invalid}" if invalid else "No valid interaction provided.",
                }
            else:
                sp = safe_input(
                    f"[Turn {turn_idx}] Enter speed(s) in [0,3] (1 value or {len(signals)} values): "
                ).strip()
                try:
                    speeds = parse_speeds(sp, len(signals))
                    cmd = {"type": "run", "turn": turn_idx, "signals": signals, "speeds": speeds}
                except Exception as e:
                    cmd = {"type": "invalid", "turn": turn_idx, "err": f"Invalid speed input: {e}"}
    else:
        cmd = None

    cmd = bcast_object(cmd, src=0)
    return cmd


def main():
    setup_stdin_guard()

    rank = get_rank()

    image_path = "./data/test_case1/ref_image.png"
    input_image = Image.open(image_path).convert("RGB")

    pretrained_model_path = "tencent/Hunyuan-GameCraft-1.0"
    pipeline = HunyuanGameCraftPipeline.from_pretrained(
        model_path=pretrained_model_path,
        device="cuda",
        cpu_offload=False,
        seed=250160,
    )

    AVAILABLE_INTERACTIONS = [
        "forward", "left", "right", "backward",
        "camera_l", "camera_r", "camera_up", "camera_down",
    ]

    interaction_text_prompt = "A charming medieval village with cobblestone streets, thatched-roof houses."
    interaction_positive_prompt = "Realistic, High-quality."
    interaction_negative_prompt = (
        "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, "
        "bad limbs, distortion, blurring, text, subtitles, static, picture, black border."
    )

    output_H = 704
    output_W = 1216
    fps = 24
    quality = 8

    cfg_scale = 2.0
    infer_steps = 50
    flow_shift_eval_video = 5.0

    if rank == 0:
        print("Available interactions:")
        for i, it in enumerate(AVAILABLE_INTERACTIONS):
            print(f"  {i + 1}. {it}")
        print("Tips:")
        print("  - input multiple interactions separated by comma (e.g., 'backward,camera_l')")
        print("  - then input speeds: either one number (broadcast) or same count (e.g., '0.2,0.3')")
        print("  - input 'n' or 'q' to stop and export video")
        print("--- Interactive Stream Started ---")

    turn_idx = 0
    while True:
        cmd = make_next_cmd(turn_idx, AVAILABLE_INTERACTIONS)

        if cmd["type"] == "stop":
            if rank == 0:
                print("Stopping interaction loop...")
            barrier()
            break

        if cmd["type"] == "invalid":
            if rank == 0:
                print(cmd.get("err", "Invalid input."))
                print(f"Please choose from: {AVAILABLE_INTERACTIONS}")
            barrier()
            continue

        signals = cmd["signals"]
        speeds = cmd["speeds"]

        if rank == 0:
            print(f"Processing turn {turn_idx} with signals={signals}, speeds={speeds}")

        start_img = input_image if turn_idx == 0 else None

        video_seg = pipeline.stream(
            interactions=signals,
            interaction_speed=speeds,
            images=start_img,
            prompt=interaction_text_prompt,
            interaction_positive_prompt=interaction_positive_prompt,
            interaction_negative_prompt=interaction_negative_prompt,
            size=(output_H, output_W),
            cfg_scale=cfg_scale,
            infer_steps=infer_steps,
            flow_shift_eval_video=flow_shift_eval_video,
        )

        if rank == 0:
            seg_len = 0 if video_seg is None else len(video_seg)
            total_len = len(getattr(pipeline.memory_module, "all_frames", []))
            print(f"Frames generated in this turn: {seg_len}, Total frames: {total_len}")

        turn_idx += 1
        barrier()

    if rank == 0:
        all_frames = getattr(pipeline.memory_module, "all_frames", [])
        print(f"Total frames generated: {len(all_frames)}")
        if len(all_frames) == 0:
            print("No frames to save. Exiting.")
            return
        imageio.mimsave("hunyuan_game_craft_stream_demo.mp4", all_frames, fps=fps, quality=quality)
        print("Saved to: hunyuan_game_craft_stream_demo.mp4")


if __name__ == "__main__":
    main()
