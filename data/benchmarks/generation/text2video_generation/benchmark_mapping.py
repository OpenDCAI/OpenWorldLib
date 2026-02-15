# from .videoverse.info import videoverse_info
# from .worldscore.info import worldscore_info

# benchmarks = {
#     "videoverse": videoverse_info,
#     "worldscore": worldscore_info
# }


eval_prompt = lambda sample: f'''You are an expert evaluator for text-to-video generation quality. You will be provided with a [Generated Video] and the corresponding [Text Prompt]. Please assess the generated video based on the following information.

[Text Prompt]
{sample.get("generation_text", "")}

[Evaluation Criteria]
Please evaluate the generated video on the following five dimensions (score 1-10 for each) and provide an overall assessment:

1. Text-Video Alignment: Does the generated video accurately represent the content described in the text prompt? Are the key elements, actions, objects, and scenes from the prompt clearly visible and correctly depicted in the video?

2. Visual Quality: Are the video frames clear, free of noticeable artifacts, blurring, or flickering? Is the overall image quality reasonable? Are colors, lighting, and details well-rendered?

3. Temporal Consistency: Are consecutive frames coherent and natural? Are there sudden jumps, objects appearing/disappearing, or structural inconsistencies? Does the video maintain logical continuity throughout?

4. Content Relevance: Does the video content match the semantic meaning and intent of the text prompt? Are the described actions, objects, and scenes relevant and appropriate to the prompt?

5. Motion Naturalness: Is the motion in the video smooth and natural? Are object movements, camera movements (if any), and scene transitions realistic and fluid? Are there unnatural jitters or abrupt changes?

[Output Format]
Please strictly output the evaluation result in the following format:
[Text-Video Alignment Score]: <a number between 1 and 10>
[Visual Quality Score]: <a number between 1 and 10>
[Temporal Consistency Score]: <a number between 1 and 10>
[Content Relevance Score]: <a number between 1 and 10>
[Motion Naturalness Score]: <a number between 1 and 10>
[Overall Score]: <a float between 1.0 and 10.0>
[Comments]: <string>
'''

info = {
    "input_keys": ["generation_text"],
    "output_keys": ["generated_video"],
    "metadata_path": "metadata.jsonl",
    "eval_prompt": eval_prompt
}

benchmarks = {
    "t2vgen_test": info,
}