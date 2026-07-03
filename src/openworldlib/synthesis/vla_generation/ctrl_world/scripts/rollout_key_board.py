
# from openpi.training import config as config_pi
# from openpi.policies import policy_config
# from openpi_client import image_tools


import numpy as np
from accelerate import Accelerator
import torch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from models.utils import key_board_control, get_fk_solution

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import einops
from accelerate import Accelerator
import datetime
import os
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import wandb
import json
from decord import VideoReader, cpu
import swanlab
import mediapy
import sys
from scipy.spatial.transform import Rotation as R



class agent():
    def __init__(self,args):

        # args = Args()
        args.val_model_path = args.ckpt_path
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        # load ctrl-world model
        self.model = CrtlWorld(args)
        self.model.load_state_dict(torch.load(args.val_model_path))
        self.model.to(self.accelerator.device).to(self.dtype)
        self.model.eval()
        print("load world model success")
        with open(f"{args.data_stat_path}", 'r') as f:
            data_stat = json.load(f)
            self.state_p01 = np.array(data_stat['state_01'])[None,:]
            self.state_p99 = np.array(data_stat['state_99'])[None,:]


    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def get_traj_info(self, id, start_idx=0, steps=8):
        val_dataset_dir = self.args.val_dataset_dir
        args = self.args
        skip = args.skip_step
        num_frames = steps
        annotation_path = f"{val_dataset_dir}/annotation/val/{id}.json"
        with open(annotation_path) as f:
            anno = json.load(f)
            try:
                length = len(anno['action'])
            except:
                length = anno["video_length"]
        frames_ids = np.arange(start_idx, start_idx + num_frames * skip, skip)
        max_ids = np.ones_like(frames_ids) * (length - 1)
        frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)
        print("Ground truth frames ids", frames_ids)

        # get action and joint pos
        instruction = anno['texts'][0]
        car_action = np.array(anno['states'])
        car_action = car_action[frames_ids]
        joint_pos = np.array(anno['joints'])
        joint_pos = joint_pos[frames_ids]

        # get videos
        video_dict =[]
        video_latent = []
        for id in range(len(anno['videos'])):
            video_path = anno['videos'][id]['video_path']
            video_path = f"{val_dataset_dir}/{video_path}"
            # load videos from all views
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
            try:
                true_video = vr.get_batch(range(length)).asnumpy()
            except:
                true_video = vr.get_batch(range(length)).numpy()
            true_video = true_video[frames_ids]
            video_dict.append(true_video)

            # encode video
            device = self.device
            true_video = torch.from_numpy(true_video).to(self.dtype).to(device)
            x = true_video.permute(0,3,1,2).to(device) / 255.0*2-1
            vae = self.model.pipeline.vae
            with torch.no_grad():
                batch_size = 32
                latents = []
                for i in range(0, len(x), batch_size):
                    batch = x[i:i+batch_size]
                    latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor)
                    latents.append(latent)
                x = torch.cat(latents, dim=0)

            video_latent.append(x)


        return car_action, joint_pos, video_dict, video_latent, instruction

    def forward_wm(self, action_cond, video_latent_true, video_latent_cond, his_cond=None, text=None):
        args = self.args
        image_cond = video_latent_cond

        # action should be normed
        action_cond = self.normalize_bound(action_cond, self.state_p01, self.state_p99, clip_min=-1, clip_max=1)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.device).to(self.dtype)
        assert image_cond.shape[1:] == (4, 72, 40)
        assert action_cond.shape[1:] == (args.num_frames+args.num_history, args.action_dim)


        # predict future frames
        with torch.no_grad():
            bsz = action_cond.shape[0]
            if text is not None:
                text_token = self.model.action_encoder(action_cond, text, self.model.tokenizer, self.model.text_encoder)
            else:
                text_token = self.model.action_encoder(action_cond)
            pipeline = self.model.pipeline

            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=image_cond,
                text=text_token,
                width=args.width,
                height=int(args.height*3),
                num_frames=args.num_frames,
                history=his_cond,
                num_inference_steps=args.num_inference_steps,
                decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None,
                output_type='latent',
                return_dict=False,
                frame_level_cond=True,
            )
        latents = einops.rearrange(latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3,n=1) # (B, 8, 4, 32,32)


        # decode ground truth video
        true_video = torch.stack(video_latent_true, dim=0) # (bsz, 8,32,32)
        decoded_video = []
        bsz,frame_num = true_video.shape[:2]
        true_video = true_video.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,true_video.shape[0],args.decode_chunk_size):
            chunk = true_video[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        true_video = torch.cat(decoded_video,dim=0)
        true_video = true_video.reshape(bsz,frame_num,*true_video.shape[1:])
        true_video = ((true_video / 2.0 + 0.5).clamp(0, 1)*255)
        true_video = true_video.detach().to(torch.float32).cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8) #(2,16,256,256,3)

        # decode predicted video
        decoded_video = []
        bsz,frame_num = latents.shape[:2]
        x = latents.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,x.shape[0],args.decode_chunk_size):
            chunk = x[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video,dim=0)
        videos = videos.reshape(bsz,frame_num,*videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1)*255)
        videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8)

        # concatenate true videos and video
        videos_cat = np.concatenate([true_video,videos],axis=-3) # (3, 8, 256, 256, 3)
        videos_cat = np.concatenate([video for video in videos_cat],axis=-2).astype(np.uint8)

        return videos_cat, true_video, videos, latents  # np.uint8:(3, 8, 128, 256, 3) or (3, 8, 192, 320, 3)


if __name__ == "__main__":
    from config import wm_args
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--svd_model_path', type=str, default=None)
    parser.add_argument('--clip_model_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--dataset_root_path', type=str, default=None)
    parser.add_argument('--dataset_meta_info_path', type=str, default=None)
    parser.add_argument('--dataset_names', type=str, default=None)
    parser.add_argument('--task_type', type=str, default='keyboard')
    parser.add_argument('--keyboard', type=str, default='ddcu')
    parser.add_argument('--save_dir', type=str, default=None)
    args_new = parser.parse_args()

    args = wm_args(task_type=args_new.task_type)

    def merge_args(args, new_args):
        for k, v in new_args.__dict__.items():
            if v is not None:
                args.__dict__[k] = v
        return args

    args = merge_args(args, args_new)
    args.data_stat_path = os.path.join(args.dataset_meta_info_path, "droid", "stat.json")

    # create rollout agent
    Agent = agent(args)
    interact_num = args.interact_num
    pred_step = args.pred_step
    num_history = args.num_history
    num_frames = args.num_frames
    print(f'rollout with {args.task_type}')

    action_keys = args.keyboard  # e.g., 'ddcu' for down, down, close gripper, up
    interact_num = len(action_keys)


    for val_id_i, text_i, start_idx_i in zip(args.val_id, args.instruction, args.start_idx):
        # read ground truth trajectory informations
        eef_gt, joint_pos_gt, video_dict, video_latents, instruction = Agent.get_traj_info(val_id_i, start_idx=start_idx_i, steps=int(pred_step*interact_num+8))
        text_i = instruction
        print("text_i:",instruction, "eef pose at t=0", eef_gt[0], "joint at t=0", joint_pos_gt[0])

        # create buffers and push first frames to history buffer
        predicted_latents = None
        video_to_save = []
        info_to_save = []
        his_cond = []
        his_joint = []
        his_eef = []
        first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)
        assert first_latent.shape == (1, 4, 72, 40), f"Expected first_latent shape (1, 4, 72, 40), got {first_latent.shape}"
        for i in range(Agent.args.num_history*4):
            his_cond.append(first_latent)  # (1, 4, 72, 40)
            his_joint.append(joint_pos_gt[0:1])  # (1, 7)
            his_eef.append(eef_gt[0:1])  # (1, 7)

        # interact loop
        for i in range(interact_num):
            # ground truth video
            start_id = int(i*(pred_step-1))
            end_id = start_id + pred_step
            video_latent_true = [v[start_id:end_id] for v in video_latents]

            # prepare input for policy
            joint_first = his_joint[-1][0]
            state_first = his_eef[-1][0]
            if i==0:
                video_first = [v[0] for v in video_dict]
            else:
                video_first = [v[-1] for v in video_dict_pred]
            assert joint_first.shape == (8,), f"Expected joint_first shape (8,), got {joint_first.shape}"
            assert state_first.shape == (7,), f"Expected state_first shape (7,), got {state_first.shape}"

            # forward policy
            print("################ policy forward ####################")
            # in the trajectory replay model, we use action recorded in trajetcory
            current_pose = his_eef[-1]  # (1, 7)
            cartesian_pose = key_board_control(current_pose, action_keys[i], task_id=val_id_i)  # (4, 7)
            print("cartesian space action", cartesian_pose[0]) # output xyz and gripper for debug
            print("cartesian space action", cartesian_pose[-1]) # output xyz and gripper for debug

            print("################ world model forward ################")
            print(f'traj_id:{val_id_i}, interact step: {i}/{interact_num}')
            # retrive history cond and action cond
            history_idx = [0,0,-8,-6,-4,-2]
            his_pose = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)  # (4, 7)
            action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)
            his_cond_input = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
            current_latent = his_cond[-1]  # (1, 4, 72, 40)
            assert current_latent.shape == (1, 4, 72, 40), f"Expected current_latent shape (1, 4, 72, 40), got {current_latent.shape}"
            assert action_cond.shape == (int(num_history+num_frames), 7), f"Expected action_cond shape ({int(num_history+num_frames)}, 7), got {action_cond.shape}"
            assert his_cond_input.shape == (1, int(num_history), 4, 72, 40), f"Expected his_cond_input shape (1, {int(num_history)}, 72, 40), got {his_cond_input.shape}"
            # forward world model
            videos_cat, true_videos, video_dict_pred, predicted_latents = Agent.forward_wm(action_cond, video_latent_true, current_latent, his_cond=his_cond_input,text=text_i if Agent.args.text_cond else None)

            print("################ record information ################")
            # push current step to history buffer
            his_eef.append(cartesian_pose[pred_step-1:pred_step]) #(1,7)
            his_cond.append(torch.cat([v[pred_step-1] for v in predicted_latents], dim=1).unsqueeze(0))  # (1, 4, 72, 40)
            if i == interact_num - 1:
                video_to_save.append(videos_cat)  # save all frames for the last interaction step
            else:
                video_to_save.append(videos_cat[:pred_step-1]) # last frame is the first frame of next step, so we remove it here


        # save rollout video and info with parameters
        video = np.concatenate(video_to_save, axis=0)
        task_name = args.task_name
        text_id = args.keyboard
        videos_dir = args.val_model_path.split('/')[:-1]
        videos_dir = '/'.join(videos_dir)
        uuid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_video = f"{args.save_dir}/{task_name}/video/time_{uuid}_traj_{val_id_i}_{start_idx_i}_{pred_step}_{text_id}.mp4"
        os.makedirs(os.path.dirname(filename_video), exist_ok=True)
        mediapy.write_video(filename_video, video, fps=4)
        print(f"Saving video to {filename_video}")
        print("##########################################################################")


# CUDA_VISIBLE_DEVICES=0 python rollout_replay_traj.py
