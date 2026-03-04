COSMOS_2P5_TASKS = ['img2world']

COSMOS_2P5_REGISTRY = {
    "transformer": {
        "repo_id": "nvidia/Cosmos-Predict2.5-2B",
        "allow_patterns": "base/post_trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
    },
    "text_encoder": {
        "repo_id": "nvidia/Cosmos-Reason1-7B",
    },
    "vae": {
        "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "allow_patterns": "Wan2.1_VAE.pth",
    },
}