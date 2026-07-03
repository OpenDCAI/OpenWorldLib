import base64
import io
import json

import numpy as np
from openai import OpenAI

from .general_utils import load_video

negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，阳光，明亮"


def parse_selected_indices_and_reasons(text):
    try:
        data = json.loads(text)
        if "selected_images" in data:
            selected = data.get("selected_images", [])
        elif "sorted_images" in data:
            selected = data.get("sorted_images", [])
        else:
            selected = []
        indices = [item["index"] for item in selected]
        reasons = [item["reason"] for item in selected]
        return indices, reasons
    except Exception as e:
        print("Failed to parse JSON from model output:", e)
        return [], []


def parse_image_descriptions(text):
    try:
        data = json.loads(text)
        descriptions = data.get("image_descriptions", [])
        return descriptions
    except Exception as e:
        print("Failed to parse descriptions JSON:", e)
        return []


def get_qwen_caption_format(task_type, sampled_images=None, desc_str_insert=None, wonder_num=2):
    if task_type == "env_cls":
        # return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Judge whether this panoramic image is a 'indoor' or 'outdoor' scene. Return only 'indoor' if the image shows indoor scenes, or 'outdoor' if it does not.<|im_end|>\n<|im_start|>assistant\n"
        return "Judge whether this panoramic image is a 'indoor' or 'outdoor' scene. Return only 'indoor' if the image shows indoor scenes, or 'outdoor' if it does not."
    elif task_type == "get_caption":
        return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Please describe this panoramic image in detail about 150 to 200 words.<|im_end|>\n<|im_start|>assistant\n"
    elif task_type == "describe_splitted_images":
        image_msgs = [{"type": "image", "image": f"file://{img_path}"} for img_path in sampled_images]
        text_msg = {
            "type": "text",
            "text": (
                f"You are given {len(sampled_images)} images extracted from a panoramic scene.\n"
                "Your task: For EACH image, provide a brief yet UNIQUE description that emphasizes:\n"
                "  1. Features indicating practical roaming paths or potential walkable entrances such as doors, roads, caves, passageways, or openings.\n"
                "  2. The AMOUNT and QUALITY of useful visual information present in the image that contributes to understanding the scene’s navigability.\n"
                "\n"
                "IMPORTANT instruction:\n"
                "- Treat each image INDEPENDENTLY, without referencing other images or assuming any context beyond the single image.\n"
                "- Do NOT try to relate scenes from different images or infer connections between them.\n"
                "\n"
                "Specifically:\n"
                "- If the image contains clear, navigable routes or identifiable openings, describe these in detail.\n"
                "- If the image contains doors, describe the states of these doors (opened or closed).\n"
                "- If the image mainly shows uninformative areas like blank walls, solid obstacles, or lacks any obvious paths, explicitly mention the low information content.\n"
                "- Each description should therefore convey how much useful roaming-related information the image holds.\n"
                "\n"
                "Make sure descriptions are DISTINCT across images, avoiding repetition or generic phrases.\n\n"
                f"Provide descriptions for all images from index 0 to {len(sampled_images) - 1}.\n\n"
                "IMPORTANT:\n"
                "- Output EXACTLY one JSON object as illustrated below, with NO extra text, explanation, or comments.\n"
                "- Escape all double quotes inside descriptions as \\\".\n"
                "- Represent all newlines inside descriptions as \\n (do NOT include real newline characters).\n"
                "- Ensure the JSON structure is strictly followed for automatic parsing by JSON parsers.\n"
                "- Invalid JSON or unescaped characters will cause parsing failure, so be careful.\n"
                "\n"
                "Output EXACTLY a JSON object like below (NO extra text):\n"
                "{\n"
                "  \"image_descriptions\": [\n"
                "    {\"image_number\": 1, \"index\": 0, \"description\": \"<unique description>\"},\n"
                "    {\"image_number\": 2, \"index\": 1, \"description\": \"<unique description>\"},\n"
                "    ...\n"
                "  ]\n"
                "}\n"
            )
        }
        messages_describe = [
            {
                "role": "user",
                "content": image_msgs + [text_msg]
            }
        ]

        return messages_describe

    elif task_type == "wonder_selection":
        messages_select = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Below is a detailed analysis of each image's roaming-related features derived from prior descriptions. Based on this, "
                            f"carefully select exactly {wonder_num} images that best enable a human explorer to freely roam forward "
                            "and extend their journey beyond the current scene.\n\n"

                            f"Image analyses:\n{desc_str_insert}\n\n"

                            "Selection criteria:\n"
                            "- Do NOT pick the first three images or images with continuous indices like '0,1,2'. Do NOT pick adjacent indices like '3,4' or '7,8', selecting the best one in adjacent indices instead.\n"
                            "- Prioritize images showing clearly navigable routes such as ‘OPENED' doors, roads, cave entrances, passageways, or openings.\n"
                            "- Do not choice images with 'CLOSED' doors.\n"
                            "- Focus on paths or features that realistically support walking forward, exploring new areas, or natural scene extension.\n"
                            "- Give preference to images whose descriptions convey rich, detailed visual information indicating high scene complexity and clear navigational cues.\n"
                            "- Avoid vague or generic reasoning; each explanation must provide specific, concrete details from the image.\n\n"

                            "For each selected image, provide:\n"
                            "- The image number (starting from 1) and index (starting from 0), formatted as \"Image X (Index Y)\".\n"
                            "- A rich, detailed explanation of why the image was chosen, including description of key scene elements, how they facilitate roaming or exploration, and their unique contribution compared to other images.\n\n"

                            "Present your answer strictly as:\n"
                            "{\n"
                            "  \"selected_images\": [\n"
                            "    {\"image_number\": X, \"index\": Y, \"reason\": \"<detailed reason>\"},\n"
                            "    {\"image_number\": X, \"index\": Y, \"reason\": \"<detailed reason>\"},\n"
                            "    {\"image_number\": X, \"index\": Y, \"reason\": \"<detailed reason>\"},\n"
                            "  ]\n"
                            "}\n\n"

                            "Only output the JSON, NO extra commentary or explanation."
                        )
                    }
                ]
            }
        ]

        return messages_select

    elif task_type == "view_analysis":
        return ("<|im_start|>system\n"
                "You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                "<|vision_start|><|image_pad|><|vision_end|>Please determine whether this image contains objects with prominent foreground or scenes with prominent textures. If the image only contains an empty flat ground, walls (especially solid-colored floors and walls with non-prominent textures), or consists solely of the ground and distant scenery, return 'No'. Otherwise—i.e., if it contains one or more prominent foregrounds, or the image depicts a scene with very prominent textures—return 'Yes'. You must only return 'Yes' or 'No'.<|im_end|>\n"
                "<|im_start|>assistant\n")

    else:
        raise NotImplementedError(f"task_type {task_type} not implemented")


def get_traj_caption(LLM_ADDR, LLM_PORT, MODEL_NAME, traj_path, sample_count=8):
    client = OpenAI(api_key="EMPTY", base_url=f"http://{LLM_ADDR}:{LLM_PORT}/v1")
    prompt_text = ("Please generate a descriptive caption for the provided video, focusing strictly on the visual content of the scene while ignoring rendering artifacts. "
                   "1. Initial Scene Description: Start by providing a detailed description of the first frame. Specifically analyze the landscape, the style and material of the huts/tents, "
                   "the campfire, the texture of the ground, and the distant mountain range. 2. Camera Movement & New Elements: Briefly describe the camera movement "
                   "(e.g., panning right, rotating) and mention any new landscape features or objects that enter the view as the perspective shifts (use relatively less paragraph to describe new features)."
                   " IMPORTANT CONSTRAINTS: Ignore the Artifacts: Do not describe the gray background, the gray void, the 'hollow' or missing pixels, "
                   "or the point-cloud texture (dots/lines). No Technical Terms: Do not use words like '3D render,' 'warp,' 'glitch,' or 'sparse data.' Focus only on the scenery and the atmosphere.")

    frames = load_video(traj_path)

    # 1. Uniformly sample frames
    total_frames = len(frames)
    if total_frames > sample_count:
        indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
        assert 0 in indices
        selected_frames = [frames[i] for i in indices]
    else:
        selected_frames = frames

    # 3. Build the VLM input message
    content_list = [{"type": "text", "text": prompt_text}]
    for frame in selected_frames:
        buffered = io.BytesIO()
        frame.save(buffered, format="JPEG", quality=70)
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    messages = [
        {"role": "system", "content": "You are a video annotation assistant."},
        {"role": "user", "content": content_list}
    ]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1024,  # Adjust max_tokens to keep it concise
        temperature=0.1,
        seed=1024
    )
    caption = response.choices[0].message.content.replace('\n', '').strip()

    return caption


def generate_motion_caption_from_plot(client, model_name, plot_path):
    """
    Read the trajectory visualization image (traj_vis.png) and generate a pure camera motion description.
    """
    try:
        with open(plot_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"  [Warning] Trajectory plot not found at {plot_path}")
        return ""

    # Prompt specifically for trajectory plots
    # Explain the coordinate system to the VLM so it can interpret the plot.
    prompt_text = (
        "You act as a camera operator writing a technical script for camera motion descriptions.\n"
        "Analyze the provided 2D trajectory plot (XY Plane). \n"
        "- The **Red Line** represents the smoothed camera path.\n"
        "- **X-axis**: Vertical direction in the plot. Positive X is **Forward**, Negative X is **Backward**.\n"
        "- **Y-axis**: Horizontal direction in the plot. Positive Y is **Right**, Negative Y is **Left**.\n\n"
        "Based on the Red Line's shape and direction, describe the camera movement concisely.\n"
        "Examples: 'move forward', 'move backward and curve left', 'static then move right'.\n"
        "Output ONLY the motion description string."
    )

    messages = [
        {"role": "system", "content": "You are an expert in analyzing camera trajectory plots."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]}
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=100,
            temperature=0.1  # Low temperature for objectivity
        )
        motion_caption = response.choices[0].message.content.strip()
        print(f"  [Motion Caption]: {motion_caption}")
        return motion_caption
    except Exception as e:
        print(f"  [Motion Caption Error]: {e}")
        return ""


def generate_video_caption(client, model_name, frames, scene_context="", target_info=None, motion_context="", sample_count=8):
    """
    Add the motion_context parameter to receive descriptions from the trajectory plot.
    """
    if not frames:
        return ""

    # 1. Uniformly sample frames (unchanged)
    total_frames = len(frames)
    if total_frames > sample_count:
        indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
        selected_frames = [frames[i] for i in indices]
    else:
        selected_frames = frames

    # 2. Build the dynamic prompt
    prompt_text = "A camera moves through a scene.\n"

    # --- [New] Inject Motion Caption (the core of the second-figure paradigm) ---
    if motion_context:
        prompt_text += f"**Camera Movement (Ground Truth)**: {motion_context}\n"
        prompt_text += "Use the above movement description as the factual basis for the camera's trajectory.\n\n"

    # A. Inject global scene context
    if scene_context:
        prompt_text += f"**Global Scene Context**: {scene_context}\n\n"

    # B. Inject task-specific instructions based on task type
    if target_info:
        # Target mode
        t_label = target_info.get('label', 'an object')
        t_dir = target_info.get('direction', 'unknown direction')

        prompt_text += f"**Navigation Goal**: The camera is moving towards a target: the **{t_label}**, located in the **{t_dir}** direction.\n\n"
        prompt_text += (
            f"Describe the video briefly. Combine the **Camera Movement** provided above with the visual content. "
            "Focus on the significant objects in relation to the camera's movement. "
            f"Explain how the camera, moving as described ({motion_context}), interacts with the scene to approach the **{t_label}**. "
            "At the end, clearly highlight that the camera reaches its destination."
        )

    else:
        # Wonder mode
        prompt_text += "**Navigation Mode**: Free exploration (Wonder Mode).\n\n"
        prompt_text += (
            f"Describe the video briefly. Combine the **Camera Movement** provided above ({motion_context}) with the visual content. "
            "Focus on the spatial relationships between the camera and the objects it encounters based on this trajectory. "
            "Do not describe textures, but focus on the flow of the view."
        )

    # 3. Build the VLM input message (unchanged)
    content_list = [{"type": "text", "text": prompt_text}]

    for frame in selected_frames:
        buffered = io.BytesIO()
        frame.save(buffered, format="JPEG", quality=70)
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    messages = [
        {"role": "system", "content": "You are a camera navigation assistant."},
        {"role": "user", "content": content_list}
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.1
        )
        caption = response.choices[0].message.content.strip()
        return caption
    except Exception as e:
        print(f"  [Caption Error]: {e}")
        return "Error generating caption."
