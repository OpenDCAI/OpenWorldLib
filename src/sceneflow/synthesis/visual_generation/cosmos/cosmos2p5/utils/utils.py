import os
import torch
import socket


def find_free_port() -> int:
    """Ask OS for an available TCP port and return it.

    Note: there remains a race where the port can be taken before being used.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def load_official_weights(model, official_ckpt_path):
    print(f"Loading official weights from: {official_ckpt_path}")
    official_state = torch.load(official_ckpt_path, map_location="cpu", weights_only=False)

    ignore_prefixes = ('accum_', 'pos_embedder.', 'loss.')

    mapping = {
        'blocks.': 'transformer_blocks.',
        'adaln_modulation_self_attn.1': 'norm1.linear_1',
        'adaln_modulation_self_attn.2': 'norm1.linear_2',
        'adaln_modulation_cross_attn.1': 'norm2.linear_1',
        'adaln_modulation_cross_attn.2': 'norm2.linear_2',
        'adaln_modulation_mlp.1': 'norm3.linear_1',
        'adaln_modulation_mlp.2': 'norm3.linear_2',
        'self_attn.q_proj': 'attn1.to_q',
        'self_attn.q_norm': 'attn1.norm_q',
        'self_attn.k_proj': 'attn1.to_k',
        'self_attn.k_norm': 'attn1.norm_k',
        'self_attn.v_proj': 'attn1.to_v',
        'self_attn.output_proj': 'attn1.to_out.0',
        'cross_attn.q_proj': 'attn2.to_q',
        'cross_attn.q_norm': 'attn2.norm_q',
        'cross_attn.k_proj': 'attn2.to_k',
        'cross_attn.k_norm': 'attn2.norm_k',
        'cross_attn.v_proj': 'attn2.to_v',
        'cross_attn.output_proj': 'attn2.to_out.0',
        'mlp.layer1': 'ff.net.0.proj',
        'mlp.layer2': 'ff.net.2',
        'x_embedder.proj.1': 'patch_embed.proj',
        't_embedder.1.linear_1': 'time_embed.t_embedder.linear_1',
        't_embedder.1.linear_2': 'time_embed.t_embedder.linear_2',
        't_embedding_norm': 'time_norm',
        'crossattn_proj.0': 'text_embed.0',
        'final_layer.adaln_modulation.1': 'norm_out.linear_1',
        'final_layer.adaln_modulation.2': 'norm_out.linear_2',
        'final_layer.linear': 'proj_out',
        'action_embedder_B_D.fc1': 'action_embed.fc1',
        'action_embedder_B_D.fc2': 'action_embed.fc2',
        'action_embedder_B_3D.fc1': 'action_embed_3d.fc1',
        'action_embedder_B_3D.fc2': 'action_embed_3d.fc2',
    }

    new_state_dict = {}
    for k, v in official_state.items():
        if k.endswith('_extra_state'):
            continue
            
        new_k = k
        if new_k.startswith('net.'): new_k = new_k[4:]
        if new_k.startswith('model.'): new_k = new_k[6:]

        if new_k.startswith(ignore_prefixes):
            continue

        for old, new in mapping.items():
            if old in new_k:
                new_k = new_k.replace(old, new)
        
        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
    
    if len(missing) == 0 and len(unexpected) == 0:
        print("Successfully loaded official weights on-the-fly!")
    else:
        raise RuntimeError(f"Weight loading failed.\nMissing: {missing}\nUnexpected: {unexpected}")

    return model


def get_cosmos_2b_config(mode='base'):
    config = dict(
        in_channels=17,
        out_channels=16,
        num_attention_heads=16,
        attention_head_dim=128,
        num_layers=28,
        mlp_ratio=4.0,
        text_in_channels=100352,
        text_embed_dim=1024,
        adaln_lora_dim=256,
        max_size=(128, 240, 240),
        patch_size=(1, 2, 2),
        rope_scale=(1.0, 3.0, 3.0),
        concat_padding_mask=True,
    )
    if mode == 'action':
        config.update(dict(action_dim=7))
    return config