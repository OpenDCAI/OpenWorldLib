import json
import os
from pyexpat import model
import sys
from typing import Dict
sys.path.append(".") 



from src.sceneflow.reasoning.spatial_reasoning.spatial_ladder.spatial_ladder import SpatialLadder
from src.sceneflow.reasoning.spatial_reasoning.spatial_reasoner.spatial_reasoner import SpatialReasoner
from src.sceneflow.reasoning.spatial_reasoning.cambrian_s.cambrian_s import CambrianS

model_path=""
#spatialladder=SpatialLadder.from_pretrained("")

reasoner=SpatialReasoner.from_pretrained(model_path)

image_path = "./data/test_case1/ref_image.png"
input_prompt = "An old-fashioned European village with thatched roofs on the houses."



output_text=reasoner.inference(instruction=input_prompt,image_paths=[image_path])

print(output_text)


video_path = "./data/test_video_case1/talking_man.mp4"
input_prompt = """
Question: What is the length of the longest dimension (length, width, or height) of the refrigerator, measured in centimeters?\nPlease think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions It's encouraged to include self-reflection or verification in the reasoning process. \n\nPlease provide your detailed reasoning between the <think> </think> tags, and then answer the question with a numerical value (e.g., 42 or 3.1) within the <answer> </answer> tags.
"""

output_text=reasoner.inference(video_paths=[video_path],instruction=input_prompt)

print(output_text)