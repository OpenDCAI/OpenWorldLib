import json
import os
import sys
from typing import Dict
sys.path.append("..") 

import requests

from src.sceneflow.pipelines.veo.pipeline_veo3 import Veo3Pipeline


image_path = "./data/test_case1/ref_image.png"
input_prompt = "An old-fashioned European village with thatched roofs on the houses."

veo3_pipeline = Veo3Pipeline.api_init(
    endpoint='https://api.newcoin.top/v1',
    api_key='sk-xtIk2ZVUneGeXmr7DkvxGDKnfiwIIJvS1xKvQ8X9nFBfbNxh')

result = veo3_pipeline(
    image=image_path,
    prompt=input_prompt
)

print(result)

# download video from result 由于目前仅支持三方api，暂时没有实现统一的下载路径