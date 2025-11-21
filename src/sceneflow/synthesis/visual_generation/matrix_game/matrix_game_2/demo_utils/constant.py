
import torch

base_size = 80
base_size2 = 44
ZERO_VAE_CACHE = [
    torch.zeros(1, 16, 2, base_size2, base_size),
    torch.zeros(1, 384, 2, base_size2, base_size),
    torch.zeros(1, 384, 2, base_size2, base_size),
    torch.zeros(1, 384, 2, base_size2, base_size),
    torch.zeros(1, 384, 2, base_size2, base_size),
    torch.zeros(1, 384, 2, base_size2, base_size),
    torch.zeros(1, 384, 2, base_size2, base_size),
    torch.zeros(1, 384, 2, base_size2, base_size),
    torch.zeros(1, 384, 2, base_size2, base_size),
    torch.zeros(1, 384, 2, base_size2, base_size),
    torch.zeros(1, 384, 2, base_size2, base_size),
    torch.zeros(1, 384, 2, base_size2, base_size),
    torch.zeros(1, 192, 2, base_size2*2, base_size*2),
    torch.zeros(1, 384, 2, base_size2*2, base_size*2),
    torch.zeros(1, 384, 2, base_size2*2, base_size*2),
    torch.zeros(1, 384, 2, base_size2*2, base_size*2),
    torch.zeros(1, 384, 2, base_size2*2, base_size*2),
    torch.zeros(1, 384, 2, base_size2*2, base_size*2),
    torch.zeros(1, 384, 2, base_size2*2, base_size*2),
    torch.zeros(1, 192, 2, base_size2*4, base_size*4),
    torch.zeros(1, 192, 2, base_size2*4, base_size*4),
    torch.zeros(1, 192, 2, base_size2*4, base_size*4),
    torch.zeros(1, 192, 2, base_size2*4, base_size*4),
    torch.zeros(1, 192, 2, base_size2*4, base_size*4),
    torch.zeros(1, 192, 2, base_size2*4, base_size*4),
    torch.zeros(1, 96, 2, base_size2*8, base_size*8),
    torch.zeros(1, 96, 2, base_size2*8, base_size*8),
    torch.zeros(1, 96, 2, base_size2*8, base_size*8),
    torch.zeros(1, 96, 2, base_size2*8, base_size*8),
    torch.zeros(1, 96, 2, base_size2*8, base_size*8),
    torch.zeros(1, 96, 2, base_size2*8, base_size*8),
    torch.zeros(1, 96, 2, base_size2*8, base_size*8)
]

feat_names = [f"vae_cache_{i}" for i in range(len(ZERO_VAE_CACHE))]
ALL_INPUTS_NAMES = ["z", "use_cache"] + feat_names
