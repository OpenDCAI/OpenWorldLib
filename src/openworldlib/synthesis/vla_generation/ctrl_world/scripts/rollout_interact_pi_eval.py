
from openpi.training import config as config_pi
from openpi.policies import policy_config
from openpi_client import image_tools
# from openpi.shared import download

import numpy as np


from accelerate import Accelerator
import torch
from diffusers import StableVideoDiffusionPipeline
import numpy as np
# import cv2
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

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from models.utils import key_board_control, get_fk_solution


class agent():
    def __init__(self,args):

        # args = Args()
        args.val_model_path = args.ckpt_path
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        # load pi policy
        if 'pi05' in args.policy_type:
            config = config_pi.get_config("pi05_droid")
        elif 'pi0fast' in args.policy_type:
            config = config_pi.get_config("pi0fast_droid")
        elif 'pi0' in args.policy_type:
            config = config_pi.get_config("pi0_droid")
        else:
            raise ValueError(f"Unknown policy type: {args.policy_type}")
        self.policy = policy_config.create_trained_policy(config, args.pi_ckpt)

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

        # Since the official Pi-Droid model output joint velocity, and crtl-world is train on cartesian space, we need to load an light-weight adapter to transform joint velocity action into cartesian pose action.
        if args.action_adapter is not None:
            from models.action_adapter.train2 import Dynamics
            self.dynamics_model = Dynamics(action_dim=7, action_num=15, hidden_size=512).to(self.device)
            self.dynamics_model.load_state_dict(torch.load(args.action_adapter, map_location=self.device))

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


    def get_traj_info(self, id, start_idx=0, steps=8,skip=1):
        val_dataset_dir = self.args.val_dataset_dir
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
        # action_cond, video_latent_true, current_latent, his_cond=his_latent,text=text_i
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

    def forward_policy(self, videos, state, joints, text, time_step=1):

        # inference policy
        image1 = videos[1]
        image2 = videos[2]
        image1 = torch.from_numpy(image1).to(torch.uint8)  # convert to torch tensor
        image2 = torch.from_numpy(image2).to(torch.uint8)  # convert to torch tensor
        assert image1.shape == (192, 320, 3), "Image 1 shape should be (192, 320, 3), got {}".format(image1.shape)
        image1 = torch.nn.functional.interpolate(image1.permute(2, 0, 1).unsqueeze(0).float(), size=(180, 320), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0).to(torch.uint8)
        image2 = torch.nn.functional.interpolate(image2.permute(2, 0, 1).unsqueeze(0).float(), size=(180, 320), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0).to(torch.uint8)
        image1 = image1.numpy()  # convert back to numpy array
        image2 = image2.numpy()  # convert back to numpy array
        example = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(image1, 224, 224),
            "observation/wrist_image_left": image_tools.resize_with_pad(image2, 224, 224),
            "observation/joint_position": joints[:7],
            "observation/gripper_position": joints[-1:],
            "prompt": text,
        }
        action_chunk = self.policy.infer(example)["actions"] #(10,8) velocity

        # action adapater
        current_joint = joints[None,:][:,:7]
        current_gripper = joints[None,:][:,7:]
        if 'pi05' in self.args.policy_type:
            idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]  # for dynamics model, we need 15 steps
        else:
            idx = [0,1,2,3,4,5,6,7,8,9,9,9,9,9,9]
        # policy output joint velocity and gripper position
        joint_vel = action_chunk[:,:7] # (15, 7)
        gripper_pos = action_chunk[:,7:] # (15, 1)
        joint_vel = joint_vel[idx]  # (15, 7)
        gripper_pos = gripper_pos[idx]  # (15, 1)
        gripper_max = self.args.gripper_max
        z_min = self.args.z_min


        gripper_pos = np.clip(gripper_pos, 0, gripper_max)
        # calculate future joint positions
        joint_pos = self.dynamics_model(current_joint, joint_vel,None, training=False)
        # fk
        state_fk = []
        joint_pos = np.concatenate([current_joint, joint_pos], axis=0)[:15]  # (15, 7)
        gripper_pos = np.concatenate([current_gripper, gripper_pos], axis=0)[:15]  # (15, 1)
        joint_vel = joint_vel  # (15, 7)
        for i in range(joint_pos.shape[0]):
            current_state_fk = get_fk_solution(joint_pos[i,:7])
            xyz = current_state_fk[:3, 3]
            # clip z axis to avoid collision with table
            xyz[2] = np.clip(xyz[2], z_min, None)
            rotation_matrix = current_state_fk[:3, :3]
            r = R.from_matrix(rotation_matrix)
            euler = r.as_euler('xyz')
            state_fk.append(np.concatenate([xyz, euler, gripper_pos[i]], axis=0))
        state_fk = np.array(state_fk) # (15,7)

        # prepare output
        skip = self.args.policy_skip_step
        valid_num = int(skip*(self.args.pred_step-1))
        policy_in_out = {
            'joint_pos': joint_pos[:valid_num],  # (12, 7)
            'joint_vel': joint_vel[:valid_num],  # (12, 7)
            'state_fk': state_fk[:valid_num],  # (12, 7)
        }
        state_fk_skip = state_fk[::skip][:self.args.pred_step]  # (5, 7)
        joint_pos_skip = joint_pos[::skip][:self.args.pred_step]  # (5, 7)
        joint_pos_skip = np.concatenate([joint_pos_skip, state_fk_skip[:,-1:]], axis=-1) # (5, 8) add gripper pos

        return policy_in_out, joint_pos_skip, state_fk_skip


if __name__ == "__main__":
    from config_eval import wm_args
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--svd_model_path', type=str, default=None)
    parser.add_argument('--clip_model_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--dataset_root_path', type=str, default=None)
    parser.add_argument('--dataset_meta_info_path', type=str, default=None)
    parser.add_argument('--dataset_names', type=str, default=None)
    parser.add_argument('--task_type', type=str, default=None)
    parser.add_argument('--pi_ckpt', type=str, default=None)
    args_new = parser.parse_args()

    args = wm_args(task_type=args_new.task_type)

    def merge_args(cfg, cli_args):
        for k, v in vars(cli_args).items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    args = merge_args(args, args_new)
    args.data_stat_path = os.path.join(args.dataset_meta_info_path, "droid", "stat.json")

    # create agent
    Agent = agent(args)
    interact_num = args.interact_num
    pred_step = args.pred_step
    num_history = args.num_history
    num_frames = args.num_frames
    history_idx = args.history_idx

    # run len(val_id) trajectory
    for val_id_i, text_i, start_idx_i in zip(args.val_id, args.instruction, args.start_idx):

        # get initial state and groud truth
        id = val_id_i
        eef_gt, joint_pos_gt, video_dict, video_latents,_ = Agent.get_traj_info(val_id_i, start_idx=start_idx_i, steps=int(pred_step*interact_num+8))
        print("text_i:",text_i, "eef pose at t=0", eef_gt[0], "joint at t=0", joint_pos_gt[0])

        # initialize all history buffer
        video_to_save, info_to_save = [], []
        his_cond, his_joint, his_eef = [], [], []
        first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)
        assert first_latent.shape == (1, 4, 72, 40), f"Expected first_latent shape (1, 4, 72, 40), got {first_latent.shape}"
        for i in range(Agent.args.num_history*4):
            his_cond.append(first_latent)  # (1, 4, 72, 40)
            his_joint.append(joint_pos_gt[0:1])  # (1, 7)
            his_eef.append(eef_gt[0:1])  # (1, 7)
        video_dict_pred = [v[0:1] for v in video_dict]


        # start rollout
        for i in range(interact_num):
            # get ground truth video latents
            # video_latent_true = [v[int(i*pred_step):int(i*pred_step+num_frames)] for v in video_latents]
            start_id = int(i*(pred_step-1))
            end_id = start_id + pred_step
            video_latent_true = [v[start_id:end_id] for v in video_latents]

            print("################ policy forward ####################")
            # prepare input for policy
            current_joint = his_joint[-1][0] # (1, 8)
            current_pose = his_eef[-1][0] # (1, 8)
            current_obs = [v[-1] for v in video_dict_pred]
            # forward policy
            policy_in_out, joint_pos, cartesian_pose= Agent.forward_policy(current_obs, current_pose, current_joint, text=text_i)
            print("cartesian space action", cartesian_pose[0]) # output xyz and gripper for debug
            print("cartesian space action", cartesian_pose[-1]) # output xyz and gripper for debug

            print("################ world model forward ################")
            # prepare input for world model
            print(f'task: {text_i}, traj_id: {val_id_i}, interact step: {i}/{interact_num}')
            # history_idx = [0,0,-12,-9,-6,-3]
            history_idx = args.history_idx
            action_cond = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)
            action_cond = np.concatenate([action_cond, cartesian_pose], axis=0) # (num_history+num_frames, 7)
            his_latent = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
            current_latent = his_cond[-1]  # (1, 4, 72, 40)
            # forward world model
            videos_cat, true_videos, video_dict_pred, predict_latents = Agent.forward_wm(action_cond, video_latent_true, current_latent, his_cond=his_latent,text=text_i if Agent.args.text_cond else None)

            print("################ record information ################")
            # push current step to history buffer
            his_joint.append(joint_pos[pred_step-1][None,:])  # (1, 8)
            his_eef.append(cartesian_pose[pred_step-1][None,:]) # (1, 7)
            his_cond.append(torch.cat([v[pred_step-1] for v in predict_latents], dim=1).unsqueeze(0))  # (1, 4, 72, 40)
            video_to_save.append(videos_cat[:pred_step-1])
            info_to_save.append(policy_in_out)  # save policy output info


        # save rollout video and info with parameters
        print("##########################################################################")
        video = np.concatenate(video_to_save, axis=0)
        text_id = text_i.replace(' ', '_').replace(',', '').replace('.', '').replace('\'', '').replace('\"', '')[:40]
        uuid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_video = f"{args.save_dir}/{args.task_name}/video/{args.task_type}_time_{uuid}_traj_{val_id_i}_{start_idx_i}_{args.policy_skip_step}_{text_id}.mp4"
        os.makedirs(os.path.dirname(filename_video), exist_ok=True)
        mediapy.write_video(filename_video, video, fps=4)
        print(f"Saving video to {filename_video}")
        info = {'success': 1, 'start_idx': 0, 'end_idx': video.shape[0]-1, 'instructions':text_i}
        for key in info_to_save[0].keys():
            info[key] = []
            for i in range(len(info_to_save)):
                info[key]+=info_to_save[i][key].tolist()
        # save to json
        filename_info = f"{args.save_dir}/{args.task_name}/info/{args.task_type}_time_{uuid}_traj_{val_id_i}_{start_idx_i}_{pred_step}_{text_id}.json"
        os.makedirs(os.path.dirname(filename_info), exist_ok=True)
        with open(filename_info, 'w') as f:
            json.dump(info, f, indent=4)
        print(f"Saving trajectory info to {filename_info}")
        print("##########################################################################")


# CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python rollout_interact_pi.py --task_type pickplace


# CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/rollout_interact_pi.py --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset --task_type pickplace
