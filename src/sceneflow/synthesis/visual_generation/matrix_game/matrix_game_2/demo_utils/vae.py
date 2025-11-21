from typing import List
from einops import rearrange
import tensorrt as trt
import torch
import torch.nn as nn

from demo_utils.constant import ALL_INPUTS_NAMES, ZERO_VAE_CACHE
from wan.modules.vae import AttentionBlock, CausalConv3d, RMS_norm, Upsample

CACHE_T = 2


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache_1, feat_cache_2):
        h = self.shortcut(x)
        feat_cache = feat_cache_1
        out_feat_cache = []
        for layer in self.residual:
            if isinstance(layer, CausalConv3d):
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache)
                out_feat_cache.append(cache_x)
                feat_cache = feat_cache_2
            else:
                x = layer(x)
        return x + h, *out_feat_cache


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, is_first_frame, feat_cache):
        if self.mode == 'upsample3d':
            b, c, t, h, w = x.size()
            # x, out_feat_cache = torch.cond(
            #     is_first_frame,
            #     lambda: (torch.cat([torch.zeros_like(x), x], dim=2), feat_cache.clone()),
            #     lambda: self.temporal_conv(x, feat_cache),
            # )
            # x, out_feat_cache = torch.cond(
            #     is_first_frame,
            #     lambda: (torch.cat([torch.zeros_like(x), x], dim=2), feat_cache.clone()),
            #     lambda: self.temporal_conv(x, feat_cache),
            # )
            x, out_feat_cache = self.temporal_conv(x, is_first_frame, feat_cache)
            out_feat_cache = torch.cond(
                is_first_frame,
                lambda: feat_cache.clone().contiguous(),
                lambda: out_feat_cache.clone().contiguous(),
            )
            # if is_first_frame:
            #     x = torch.cat([torch.zeros_like(x), x], dim=2)
            #     out_feat_cache = feat_cache.clone()
            # else:
            #     x, out_feat_cache = self.temporal_conv(x, feat_cache)
        else:
            out_feat_cache = None
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x, out_feat_cache

    def temporal_conv(self, x, is_first_frame, feat_cache):
        b, c, t, h, w = x.size()
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache is not None:
            cache_x = torch.cat([
                torch.zeros_like(cache_x),
                cache_x
            ], dim=2)
        x = torch.cond(
            is_first_frame,
            lambda: torch.cat([torch.zeros_like(x), x], dim=1).contiguous(),
            lambda: self.time_conv(x, feat_cache).contiguous(),
        )
        # x = self.time_conv(x, feat_cache)
        out_feat_cache = cache_x

        x = x.reshape(b, 2, c, t, h, w)
        x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                        3)
        x = x.reshape(b, c, t * 2, h, w)
        return x.contiguous(), out_feat_cache.contiguous()

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        # conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        # init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class VAEDecoderWrapperSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = VAEDecoder3d()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.z_dim = 16
        self.conv2 = CausalConv3d(self.z_dim, self.z_dim, 1)

    def forward(
            self,
            z: torch.Tensor,
            is_first_frame: torch.Tensor,
            *feat_cache: List[torch.Tensor]
    ):
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        z = z.permute(0, 2, 1, 3, 4)
        assert z.shape[2] == 1
        feat_cache = list(feat_cache)
        is_first_frame = is_first_frame.bool()

        device, dtype = z.device, z.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        x = self.conv2(z)
        out, feat_cache = self.decoder(x, is_first_frame, feat_cache=feat_cache)
        out = out.clamp_(-1, 1)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        out = out.permute(0, 2, 1, 3, 4)
        return out, feat_cache


class VAEDecoder3d(nn.Module):
    def __init__(self,
                 dim=96,
                 z_dim=16,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        self.cache_t = 2
        self.decoder_conv_num = 32

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(
            self,
            x: torch.Tensor,
            is_first_frame: torch.Tensor,
            feat_cache: List[torch.Tensor]
    ):
        idx = 0
        out_feat_cache = []

        # conv1
        cache_x = x[:, :, -self.cache_t:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat([
                feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                    cache_x.device), cache_x
            ],
                dim=2)
        x = self.conv1(x, feat_cache[idx])
        out_feat_cache.append(cache_x)
        idx += 1

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x, out_feat_cache_1, out_feat_cache_2 = layer(x, feat_cache[idx], feat_cache[idx + 1])
                idx += 2
                out_feat_cache.append(out_feat_cache_1)
                out_feat_cache.append(out_feat_cache_2)
            else:
                x = layer(x)

        # upsamples
        for layer in self.upsamples:
            if isinstance(layer, Resample):
                x, cache_x = layer(x, is_first_frame, feat_cache[idx])
                if cache_x is not None:
                    out_feat_cache.append(cache_x)
                    idx += 1
            else:
                x, out_feat_cache_1, out_feat_cache_2 = layer(x, feat_cache[idx], feat_cache[idx + 1])
                idx += 2
                out_feat_cache.append(out_feat_cache_1)
                out_feat_cache.append(out_feat_cache_2)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                cache_x = x[:, :, -self.cache_t:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                out_feat_cache.append(cache_x)
                idx += 1
            else:
                x = layer(x)
        return x, out_feat_cache


class VAETRTWrapper():
    def __init__(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open("checkpoints/vae_decoder_int8.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            self.engine: trt.ICudaEngine = rt.deserialize_cuda_engine(f.read())

        self.context: trt.IExecutionContext = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream().cuda_stream

        # ──────────────────────────────
        # 2️⃣  Feed the engine with tensors
        #     (name-based API in TRT ≥10)
        # ──────────────────────────────
        self.dtype_map = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int8: torch.int8,
            trt.int32: torch.int32,
        }
        test_input = torch.zeros(1, 16, 1, 60, 104).cuda().half()
        is_first_frame = torch.tensor(1.0).cuda().half()
        test_cache_inputs = [c.cuda().half() for c in ZERO_VAE_CACHE]
        test_inputs = [test_input, is_first_frame] + test_cache_inputs

        # keep references so buffers stay alive
        self.device_buffers, self.outputs = {}, []

        # ---- inputs ----
        for i, name in enumerate(ALL_INPUTS_NAMES):
            tensor, scale = test_inputs[i], 1 / 127
            tensor = self.quantize_if_needed(tensor, self.engine.get_tensor_dtype(name), scale)

            # dynamic shapes
            if -1 in self.engine.get_tensor_shape(name):
                # new API :contentReference[oaicite:0]{index=0}
                self.context.set_input_shape(name, tuple(tensor.shape))

            # replaces bindings[] :contentReference[oaicite:1]{index=1}
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
            self.device_buffers[name] = tensor                             # keep pointer alive

        # ---- (after all input shapes are known) infer output shapes ----
        # propagates shapes :contentReference[oaicite:2]{index=2}
        self.context.infer_shapes()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            # replaces binding_is_input :contentReference[oaicite:3]{index=3}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = self.dtype_map[self.engine.get_tensor_dtype(name)]
                out = torch.empty(shape, dtype=dtype, device="cuda").contiguous()

                self.context.set_tensor_address(name, int(out.data_ptr()))
                self.outputs.append(out)
                self.device_buffers[name] = out

    # helper to quant-convert on the fly
    def quantize_if_needed(self, t, expected_dtype, scale):
        if expected_dtype == trt.int8 and t.dtype != torch.int8:
            t = torch.clamp((t / scale).round(), -128, 127).to(torch.int8).contiguous()
        return t                            # keep pointer alive

    def forward(self, *test_inputs):
        for i, name in enumerate(ALL_INPUTS_NAMES):
            tensor, scale = test_inputs[i], 1 / 127
            tensor = self.quantize_if_needed(tensor, self.engine.get_tensor_dtype(name), scale)
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
            self.device_buffers[name] = tensor

        self.context.execute_async_v3(stream_handle=self.stream)
        torch.cuda.current_stream().synchronize()
        return self.outputs
