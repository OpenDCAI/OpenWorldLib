python pipeline.py \
    --pretrained-model-name-or-path /path/to/HY-Pano-2.0 \
    --subfolder "" \
    --image /path/to/input.png \
    --prompt "Expand this image to a 360-degree equirectangular panorama. Maintain realistic style." \
    --save temp/out_panorama.png

# python pipeline_with_qwen_image.py \
#     --lora-path /path/to/lora \
#     --lora-subfolder "" \
#     --image /path/to/input.png \
#     --prompt "A sunny outdoor scene." \
#     --save temp/out_panorama.png
