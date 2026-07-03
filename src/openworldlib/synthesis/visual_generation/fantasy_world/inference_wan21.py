# Copyright Alibaba Inc. All Rights Reserved.
import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import json
import random
import sys
from pathlib import Path
from PIL import Image
import numpy as np

import torch

from FantasyWorld.fusion.model_wan21 import FantasyWorldFusionModel
from openworldlib.base_models.diffusion_model.diffsynth.data.dataset_re10k import RealEstate10KPoseProcessor
from openworldlib.base_models.three_dimensions.point_clouds.vggt.vggt.utils.pose_enc import extri_intri_to_pose_encoding
from utils import cameras_json_to_camera_list, get_pointclouds, save_colored_pointcloud_ply, normalize_scene, get_intrinsic_matrix, batch_depth_to_world, save_video_imageio

try:
    from moge.model.v2 import MoGeModel
except ImportError:
    from openworldlib.base_models.three_dimensions.depth.moge.model.v2 import MoGeModel_v2 as MoGeModel

def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(
        description="FantasyWorld Stage 2 Inference")
    parser.add_argument(
        "--wan_ckpt_path",
        type=str,
        required=True,
        help="ckpt path")
    parser.add_argument(
        "--model_ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint file (.pth)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        default="examples/images/input_image.png",
        help="Path to input image")
    parser.add_argument(
        "--camera_json_path",
        type=str,
        required=True,
        default="examples/cameras/camera_data.json",
        help="Path to camera parameters JSON file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        default="In the Open Loft Living Room, sunlight streams through large windows, highlighting the sleek fireplace and elegant wooden stairs.",
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default=(
            "Bright tones, overexposed, static, blurred details, subtitles, style, "
            "works, paintings, images, static, overall gray, worst quality, low quality, "
            "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
            "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
            "still picture, messy background, three legs, many people in the background, "
            "walking backwards"),
        help="Negative prompt for generation")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated video"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Frames per second"
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=50,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="Random seed"
    )
    parser.add_argument(
        "--using_scale",
        type=str2bool,
        default=True,
        help="Whether to use scale normalization (True/False)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=336,
        help="Video height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=592,
        help="Video width"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=17,
        help="Number of frames to generate"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=1.0,
        help="Confidence threshold for saving point clouds"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Stride for saving point clouds"
    )
    args = parser.parse_args()
    return args


class FantasyWorldSampler:
    def __init__(
        self,
        sample_steps=40,
        sample_guide_scale=5.0,
        size="832*480",
        ckpt_dir="./models/Wan2.1-I2V-14B-480P",
        model_ckpt=None,
        frames=81,
        fps=16,
        height=336,
        width=592,
        start_index=16,
    ):
        # Initialize your model
        self.sample_steps = sample_steps
        self.sample_guide_scale = sample_guide_scale
        self.size = size
        self.fps = fps
        self.device = "cuda"
        self.torch_dtype = torch.bfloat16
        self.num_frames = frames
        self.height = height
        self.width = width
        self.start_index = start_index

        print("Creating WanI2V pipeline.")
        self.pose_processor = RealEstate10KPoseProcessor(
            sample_stride=1,
            sample_n_frames=frames,
            relative_pose=True,
            zero_t_first_frame=True,
            sample_size=[height, width],
            rescale_fxy=False,
            shuffle_frames=False,
            use_flip=False,
            is_i2v=True,
        )
        dit_path = [[f"{ckpt_dir}/diffusion_pytorch_model-0000{i}-of-00007.safetensors"
                     for i in range(1, 8)],
                    f"{ckpt_dir}/Wan2.1_VAE.pth",
                    f"{ckpt_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                    f"{ckpt_dir}/models_t5_umt5-xxl-enc-bf16.pth",
                    ]
        vggt_cfg = {
            "enable_camera": True,
            "enable_depth": True,
            "enable_point": True,
            "enable_track": False,
        }

        camera_cfg = {
            "pose_in_dim": 768,
            "plucker_fea_dim": 2048,
            "pose_inject_method": "adaln",
            "use_info": "plucker",
        }

        self.model = FantasyWorldFusionModel(
            start_index=self.start_index,
            use_gradient_checkpointing=True,
            cross_attention_list=list(range(24)),
            dit_path=dit_path,
            vggt_cfg=vggt_cfg,
            camera_control=True,
            camera_cfg=camera_cfg,
        )

        # Load model checkpoint if provided
        if model_ckpt is not None:
            print(f"Loading model checkpoint from: {model_ckpt}")
            ckpt = torch.load(model_ckpt, map_location="cpu")
            messages = self.model.load_state_dict(ckpt, strict=False)
            print("Missing keys = {}, Unexpected keys = {}".format(len(messages.missing_keys), len(messages.unexpected_keys)))
            if messages.unexpected_keys:
                print("Unexpected key examples: {}".format(messages.unexpected_keys[:10]))
        else:
            print("No model checkpoint provided, using uninitialized model")
        moge_ckpt = os.environ.get("FANTASY_WORLD_MOGE_CKPT", "Ruicheng/moge-2-vitl-normal")
        self.moge = MoGeModel.from_pretrained(moge_ckpt).to(self.device).eval()
        self.model.to(torch.bfloat16)
        self.model.to(self.device)
        self.model.pipe.device = self.device
        self.model.eval()

    def generate_video(
        self,
        prompt,
        neg_prompt,
        image_path=None,
        camera_params=None,
        using_scale=True,
        seed=1024,
    ):
        print("Generating video ...")
        with torch.no_grad():
            input_image = Image.open(image_path).convert('RGB')
            input_image = np.array(input_image)
            input_image = torch.tensor(
                input_image / 255,
                dtype=torch.float32,
                device=self.device
            ).permute(2, 0, 1)
            output = self.moge.infer(input_image)
            moge = {k: v.cpu().contiguous() for k, v in output.items()}
            intrinsics = []
            extrinsics = []
            for camera in camera_params:
                intrinsics.append(get_intrinsic_matrix(camera))
                extrinsics.append(camera.w2c_mat)
            intrinsics = torch.from_numpy(
                np.stack(intrinsics).astype(np.float32))
            extrinsics = torch.from_numpy(
                np.stack(extrinsics).astype(np.float32))
            extrinsics_4x4 = extrinsics.unsqueeze(0)

            if using_scale:
                first_intrinsic = intrinsics[0, :, :].unsqueeze(0)
                first_extrinsic = extrinsics[0, :3, :].unsqueeze(0)

                first_moge_world, first_moge_mask = batch_depth_to_world(
                    prediction=moge,
                    extrinsics=first_extrinsic,
                    intrinsics=first_intrinsic
                )
                extrinsics_3x4 = extrinsics_4x4[:, :, :3, :]
                extrinsics = normalize_scene(
                    extrinsics=extrinsics_3x4,
                    first_moge_world=first_moge_world.unsqueeze(0),
                    first_moge_mask=first_moge_mask.unsqueeze(0),
                ).squeeze(0)

            image_hw = [self.height, self.width]

            pose_enc = extri_intri_to_pose_encoding(
                extrinsics.unsqueeze(0),
                intrinsics.unsqueeze(0),
                image_hw,
                pose_encoding_type="absT_quaR_FoV"
            ).squeeze(0)
            plucker_embedding = self.pose_processor.get_plucker_embedding_direct_from_cam_params(
                pose_enc.unsqueeze(0), image_size=(
                    self.height, self.width)).to(
                self.device, self.torch_dtype)

            end_image = None
            first_frame_image = Image.open(image_path).convert('RGB')
            image_emb = self.model.pipe.encode_image(
                first_frame_image,
                end_image,
                self.num_frames,
                self.height,
                self.width
            )
            clip_feature = image_emb['clip_feature'].to(
                self.device, self.torch_dtype)
            y = image_emb['y'].to(self.device, self.torch_dtype)
            ctx_pos = self.model.pipe.encode_prompt(prompt)['context'].to(
                self.device,
                self.torch_dtype
            )
            ctx_neg = self.model.pipe.encode_prompt(neg_prompt)['context'].to(
                self.device,
                self.torch_dtype
            )

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.torch_dtype):
            latent_video, prediction = self.model.generate_video(
                context_pos=ctx_pos,
                context_neg=ctx_neg,
                clip_feature=clip_feature,
                y=y,
                height=self.height,
                width=self.width,
                num_inference_steps=self.sample_steps,
                num_frames=self.num_frames,
                image_path=image_path,
                plucker_embedding=plucker_embedding,
                seed=seed
            )
            frames = self.model.pipe.vae.decode(
                latent_video,
                device=self.device,
                tiled=True,
                tile_size=(30, 52),
                tile_stride=(15, 26)
            )
            video = frames.squeeze(0)
            video = video.permute(1, 2, 3, 0)
            video = video.to(torch.float32).cpu()
            video = (video + 1.0) / 2.0
            video = (video * 255.0).clamp(0, 255)
            frames_np_processed = video.numpy().astype(np.uint8)
        return frames_np_processed, prediction


def main():
    args = parse_args()
    camera_json_path = args.camera_json_path

    if not os.path.exists(camera_json_path):
        raise FileNotFoundError(
            f"Camera data file not found: {camera_json_path}")

    print(f"Loading camera data from: {camera_json_path}")
    with open(camera_json_path, 'r') as f:
        camera_data = json.load(f)

    camera_params = cameras_json_to_camera_list(camera_data, image_size=(args.height, args.width))

    print("Initializing FantasyWorld model...")
    model = FantasyWorldSampler(
        sample_steps=args.sample_steps,
        fps=args.fps,
        ckpt_dir=args.wan_ckpt_path,
        model_ckpt=args.model_ckpt,
        height=args.height,
        width=args.width,
        frames=args.frames,
    )

    # Inference
    video, prediction = model.generate_video(
        prompt=args.prompt,
        neg_prompt=args.neg_prompt,
        image_path=args.image_path,
        camera_params=camera_params,
        using_scale=args.using_scale,
        seed=args.seed,
    )
    # Save output video
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_stem = Path(args.image_path).stem
    output_path_video = output_dir / "video.mp4"

    save_video_imageio(video, output_path_video, fps=args.fps)
    print(f"Video saved to: {output_path_video}")

    recon_worldpoints = get_pointclouds(prediction, fix_first_frame=True)

    # save point clouds
    pc_path_recon = output_dir / \
        f"recon_confthresh{args.conf_threshold}.ply"

    valid_mask = prediction['depth_conf'] >= args.conf_threshold
    save_colored_pointcloud_ply(
        points=recon_worldpoints,
        colors=video,
        out_path=pc_path_recon,
        stride=args.stride,
        max_points=None,
        valid_mask=valid_mask.cpu().numpy()[0],
    )
    print(f"Point clouds have saved to: {pc_path_recon}")


if __name__ == "__main__":
    main()
