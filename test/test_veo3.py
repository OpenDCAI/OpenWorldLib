import json
import os
import sys
from typing import Dict
sys.path.append("..") 

import requests

from src.sceneflow.pipelines.veo3.pipeline_veo3 import Veo3Pipeline


image_path = "./data/test_case1/ref_image.png"
output_path = "./output/veo3/generated.mp4"
input_prompt = "An old-fashioned European village with thatched roofs on the houses."
duration_seconds = 8

veo3_pipeline = Veo3Pipeline.from_pretrained(
    api_key='sk-PDZWiPJbKEyVf6VwBRxVklZ6um9a7cmizbvF7QoQ5E7QcyCn',
    base_url='https://sg.uiuiapi.com/v1')

result = veo3_pipeline(
    image=image_path,
    prompt=input_prompt,
    duration_seconds=duration_seconds
)

print('-------------result-------------', result)