eval_prompt = lambda sample: f'''You are an expert evaluator for navigation video generation quality. You will be provided with a [Reference Image] and a [Generated Video]. Please assess the generated navigation video based on the following information.

[Navigation Parameters]
- Interaction signal sequence: {sample.get("interaction_signal", [])}
- Scene description: {sample.get("scene_description", "")}

[Evaluation Criteria]
Please evaluate the generated video on the following five dimensions (score 1-10 for each) and provide an overall assessment:

1. Navigation Fidelity: Does the generated video accurately execute the given interaction signal sequence? For example, does the camera move forward when the signal is "forward", turn left on "left", and pan the camera on "camera_l" / "camera_r"?

2. Visual Quality: Are the video frames clear, free of noticeable artifacts, blurring, or flickering? Is the overall image quality reasonable?

3. Temporal Consistency: Are consecutive frames coherent and natural? Are there sudden jumps, objects appearing/disappearing, or structural inconsistencies?

4. Scene Consistency: Does the generated video maintain consistency with the input reference image in terms of texture, style, and object layout?

5. Motion Smoothness: Is the camera motion smooth and natural? Are there unnatural jitters or abrupt speed changes?

[Output Format]
Please strictly output the evaluation result in the following format:
[Navigation Fidelity Score]: <a number between 1 and 10>
[Visual Quality Score]: <a number between 1 and 10>
[Temporal Consistency Score]: <a number between 1 and 10>
[Scene Consistency Score]: <a number between 1 and 10>
[Motion Smoothness Score]: <a number between 1 and 10>
[Overall Score]: <a float between 1.0 and 10.0>
[Comments]: <string>
'''

info = {
    "input_keys": ["ref_image", "interaction_signal", "scene_description"],
    "output_keys": ["generated_video"],
    "perception_data_path": "test_images/",
    "metadata_path": "metadata.jsonl",
    "eval_prompt": eval_prompt
}

benchmarks = {
    "sf_nav_vidgen_test": info,
}
