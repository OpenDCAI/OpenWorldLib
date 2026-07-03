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
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import random

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Dynamics(nn.Module):
    def __init__(self, action_dim, action_num, hidden_size):
        super().__init__()
        self.action_dim = action_dim
        self.action_num = action_num
        self.hidden_size = hidden_size

        self.joint_vel_01 = np.array([-0.4077107 , -0.79047304 ,-0.47850373 ,-0.8666644 , -0.6729502 , -0.5602032,-0.692411])[None,:]
        self.joint_vel_99 = np.array([0.4900636 , 0.7259861 , 0.45910007 ,0.79220384 ,0.69864315, 0.648198,0.810115])[None,:]
        self.joint_delta_01 = np.array([-0.2801219,  -0.397792,   -0.22935797, -0.3351759,  -0.42025003, -0.36825255, -0.450706])[None,:]
        self.joint_delta_99 = np.array([0.2827909,  0.42184818, 0.33529875, 0.35958457, 0.375613,0.44463825, 0.4697690])[None,:]

        # self.CLS = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
        input_dim = int(action_dim * (action_num+1))
        output_dim = int(action_num * action_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, output_dim),
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def forward(self, joint, joint_vel, joint_delta, training=True):
        # action: (B, T, action_num, action_dim)
        # action = einops.rearrange(action, 'b t d -> b 1 (t d)')
        if joint.ndim == 2:
            joint = joint[None,:]
        if joint_vel.ndim == 2:
            joint_vel = joint_vel[None,:]
        assert joint.shape[1:] == (1, self.action_dim), "Joint shape should be (B, 1, action_dim), got {}".format(joint.shape)
        assert joint_vel.shape[1:] == (self.action_num, self.action_dim), "Joint velocity shape should be (B, action_num, action_dim), got {}".format(joint_vel.shape)
        # assert joint_delta.shape[1:] == (self.action_num, 7), "Joint delta shape should be (B, action_num, 7), got {}".format(joint_delta.shape)
        joint = torch.tensor(joint).float().to(self.device)  # (B, T, action_num, action_dim)
        joint_vel = self.normalize_bound(joint_vel, np.array(self.joint_vel_01), np.array(self.joint_vel_99))
        joint_vel = torch.tensor(joint_vel).float().to(self.device)  # (B, T, action_num, action_dim)

        B = joint.shape[0]
        joint = joint.reshape(B, -1)  # (B*T, 8)
        joint_vel = joint_vel.reshape(B, -1)  # (B*T,
        input = torch.cat((joint, joint_vel), dim=1)  # (B*T, action_num*2*action_dim)
        pred = self.net(input)  # (B*T, action_num*action_dim)
        pred = einops.rearrange(pred, 'b (t d) -> b t d', t=self.action_num, d=self.action_dim)

        if training:
            joint_delta = self.normalize_bound(joint_delta, np.array(self.joint_delta_01), np.array(self.joint_delta_99))
            joint_delta = torch.tensor(joint_delta).float().to(self.device)
            loss = F.mse_loss(pred, joint_delta)  # (B, T, 7)
            return loss

        pred = pred.detach().cpu().numpy()  # (B, T, 7)
        pred = self.denormalize_bound(pred, np.array(self.joint_delta_01), np.array(self.joint_delta_99))
        joint = joint.detach().cpu().numpy()  # (B, T, action_num, action_dim)
        joint_future = joint + pred  # (B, T, action_num, action_dim)

        return joint_future[0]  # (B, action_num, action_dim)


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
        # return np.clip(ndata, clip_min, clip_max)
        return ndata
    def denormalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps=1e-8,
    ) -> np.ndarray:
        clip_range = clip_max - clip_min
        rdata = (data - clip_min) / clip_range * (data_max - data_min) + data_min
        return rdata
    def inference(self, state, action):
        """
        state: (B, 1, 7)
        action: (B, action_num, action_dim)
        """
        state = state[None,:]  # (1, 1, 7)
        assert state.shape == (1,7), "State shape should be (1, 7), got {}".format(state.shape)
        assert action.shape == (self.action_num, self.action_dim), "Action shape should be ({}, {}), got {}".format(self.action_num, self.action_dim, action.shape)

        # state = self.normalize_bound(state, np.array(self.state_01), np.array(self.state_99))
        # action = self.normalize_bound(action, np.array(self.action_01), np.array(self.action_99))
        state = torch.tensor(state).unsqueeze(0).float()  # (1, 1, 7)
        action = torch.tensor(action).unsqueeze(0).float()  # (1, action_num, action_dim
        state = state.to(self.clip_pos_emb.device)
        action = action.to(self.clip_pos_emb.device)

        with torch.no_grad():
            state_emb = self.state_encode(state)  # (1, 1, hidden_size)
            action_emb = self.action_encode(action)  # (1, action_num, hidden_size)
            input = torch.cat((state_emb, action_emb), dim=1)  # (1, action_num+1, hidden_size)
            input = input + self.clip_pos_emb
            output = self.transformer(input)  # (1, action_num+1, hidden_size)
            output = output[:,1:]  # (1, action_num, hidden_size)
            state_pred = self.state_decode(output)  # (1, action_num, 7
            state_pred = state_pred.squeeze(0).cpu().numpy()

        # state_pred = self.denormalize_bound(state_pred, np.array(self.state_01), np.array(self.state_99))
        return state_pred  # (action_num, 7)



class Dataset_xhand(Dataset):
    def __init__(
            self,
            args,
            mode = 'val',
    ):
        """Constructor."""
        super().__init__()
        self.args = args
        self.mode = mode
        data_json_path = args.data_json_path
        data_root_path = args.data_root_path

        # dataset stucture
        # dataset_dir/dataset_name/annotation_name/mode/traj
        # dataset_dir/dataset_name/video/mode/traj
        # dataset_dir/dataset_name/latent_video/mode/traj

        # samles:{'ann_file':xxx, 'frame_idx':xxx, 'dataset_name':xxx}

        # prepare all datasets path
        self.video_path = []
        data_json_path = f'{data_json_path}/{mode}_sample.json'
        with open(data_json_path, "r") as f:
            self.samples = json.load(f)
        self.video_path = [os.path.join(data_root_path, sample['dataset_name']) for sample in self.samples]

        print(f"ALL dataset, {len(self.samples)} samples in total")

        self.a_min = np.array(args.action_01)[None,:]
        self.a_max = np.array(args.action_99)[None,:]
        self.s_min = np.array(args.state_01)[None,:]
        self.s_max = np.array(args.state_99)[None,:]

    def __len__(self):
        return len(self.samples)

    def fetch(self,index):
        sample = self.samples[index]
        sampled_video_dir = self.video_path[index]
        ann_file = sample['ann_file']
        # dataset_name = sample['dataset_name']
        ann_file = f'{sampled_video_dir}/{ann_file}'
        with open(ann_file, "r") as f:
            label = json.load(f)


        traj_id = int(sample['episode_id'])
        chunk_id = int(np.floor(traj_id / 1000))

        start_id = int(sample['frame_ids'][0]*3) # random select 0,1,2
        max_id = int(sample['frame_ids'][-1]*3-3)
        max_id = np.array([max_id]* (self.args.num_frames+1))
        frame_ids_ori = np.array((range(start_id, start_id+self.args.num_frames+1)))
        frame_ids_ori = np.clip(frame_ids_ori, 0, max_id) # clip to the max id
        # print(frame_ids, frame_ids_ori)

        # load state, action
        self.old_path = os.environ.get("CTRL_WORLD_DROID_ROOT", "dataset_example/droid_subset")
        file_path = f'{self.old_path}/data/chunk-{chunk_id:03d}/episode_{traj_id:06d}.parquet'
        df = pd.read_parquet(file_path)
        joints = []
        joint_vels = []
        for i in frame_ids_ori:
            joint = df['observation.state.joint_position'][i].tolist()
            joint_vel = df['action.joint_velocity'][i].tolist()
            joints.append(joint)
            joint_vels.append(joint_vel)

        # joints = np.array(joints, dtype=np.float32)
        # joints = self.normalize_bound(joints, self.s_min, self.s_max)
        # joint_vels = np.array(joint_vels, dtype=np.float32)
        # joint_vels = self.normalize_bound(joint_vels, self.a_min, self.a_max)

        joints = np.array(joints, dtype=np.float32)
        joint_vels = np.array(joint_vels, dtype=np.float32)

        data = dict()
        data['joints'] = joints[0:1]
        data['joint_vels'] = joint_vels[:-1]
        data['joints_delta'] = joints[1:] - joints[0:1]  # (num_frames-1, 7)
        # data['file_path'] = file_path

        # if joints is None or joint_vels is None:
            # print(file_path, frame_ids_ori,)
        # print(data)
        # print(joints)
        # print(joints.shape, joint_vels.shape, file_path, frame_ids_ori)
        return data

    def __getitem__(self, index):
        try:
            data = self.fetch(index)
        except:
            print(f"Error fetching data for index {index}, retrying...")
            # print(data['file_path'])
            data = self.fetch(random.randint(0, len(self.samples)-1))

        return data


class Args:
    def __init__(self):
        self.data_json_path = 'exp_cfg/droid_svd_v3'
        self.data_root_path = os.environ.get("CTRL_WORLD_DROID_ROOT", "dataset_example/droid_subset")
        self.action_01 = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        self.action_99 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.state_01 = [-2.6157, -1.5052, -2.6296, -2.8643, -2.6102,  0.5533, -2.720]
        self.state_99 = [ 2.6244,  1.5285,  2.6203, -0.3490,  2.6432,  4.2453,  2.7410]
        self.action_dim = 7
        self.num_frames = 15
        # self.joint_vel_01 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])[None,:]
        # self.joint_vel_99 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])[None,:]
        # self.joint_delta_01 = np.array([-0.7850, -0.5931, -0.8170, -0.6413, -0.8921, -0.8346, -0.809])[None,:]
        # self.joint_delta_99 = np.array([0.7167, 0.7533, 0.7917, 0.6255, 0.9073, 0.9226, 0.861])[None,:]
        self.s_max = torch.tensor([ 2.6244,  1.5285,  2.6203, -0.3490,  2.6432,  4.2453,  2.7410]).float()
        self.s_min = torch.tensor([-2.6157, -1.5052, -2.6296, -2.8643, -2.6102,  0.5533, -2.720]).float()
        self.a_max = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).float()
        self.a_min = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]).float()


# if __name__ == "__main__":



if __name__ == "__main__":
    args = Args()
    dataset = Dataset_xhand(args, mode='train')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=16)

    # for i in range(len(dataset)):
    #     data = dataset[i]
    #     print(data['joints'].shape, data['joint_vels'].shape, data['file_path'])
        # print(data['joints'], data['joint_vels'], data['file_path'])
        # print(data)
        # break
    device = 'cuda'
    dynamics_model = Dynamics(action_dim=7, action_num=args.num_frames, hidden_size=512).to(device)
    optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=1e-4)
    update_step = 0
    loss_all = 0
    for epoch in range(10):
        for batch in tqdm(dataloader):
            joint = batch['joints']
            joint_vel = batch['joint_vels']
            joint_delta = batch['joints_delta']

            loss = dynamics_model(joint, joint_vel, joint_delta) #(B,11,7)
            # print("state_pred shape", state_pred.shape, "joints shape", joints.shape)
            # loss = F.mse_loss(state_pred, joints[:, 1:]) #(B,10,7)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_step += 1
            loss_all += loss.item()
            if update_step % 100 == 0:
                print(f'Update Step {update_step}, Loss: {loss_all / 100:.4f}')
                loss_all = 0
            # print(f'Epoch {epoch}, Loss: {loss.item()}')
        print(f'Epoch {epoch} completed')
        # save the model
        torch.save(dynamics_model.state_dict(), f'output_dynamics/model2_{args.num_frames}_{epoch}.pth')


# CUDA_VISIBLE_DEVICES=1 python output_dynamics/train2.py

    # s_max = torch.tensor([ -10,-10, -10, -10, -10, -10, -10]).float()
    # s_min = torch.tensor([ 10, 10, 10, 10, 10, 10, 10]).float()
    # a_max = torch.tensor([-10, -10, -10, -10, -10, -10, -10]).float()
    # a_min = torch.tensor([ 10, 10, 10, 10, 10, 10, 10]).float()

    # joint_deltas = []
    # joint_vel= []

    # for epoch in range(1):
    #     for batch in tqdm(dataloader):
    #         joints = batch['joints']
    #         joint_vels = batch['joint_vels']
    #         joint_delta = batch['joints_delta']

    #         joints = joints.reshape(-1, 7)  # (B*T, 8)
    #         joint_vels = joint_vels.reshape(-1, 7)  # (B
    #         joint_delta = joint_delta.reshape(-1, 7)  # (B*T, 8)

    #         joint_deltas.append(np.array(joint_delta)[-1:])  # only keep the last frame delta
    #         joint_vel.append(np.array(joint_vels)[-1:])

    # joint_deltas = np.concatenate(joint_deltas, axis=0)
    # # 1% and 99% quantile
    # s_min = np.quantile(joint_deltas, 0.01, axis=0, keepdims=True)
    # s_max = np.quantile(joint_deltas, 0.99, axis=0, keepdims=True)

    # print("State Max:", s_max)
    # print("State Min:", s_min)

    # joint_vel = np.concatenate(joint_vel, axis=0)
    # # 1% and 99% quantile
    # a_min = np.quantile(joint_vel, 0.01, axis=0, keepdims=True)
    # a_max = np.quantile(joint_vel, 0.99, axis=0, keepdims=True)
    # print("Action Max:", a_max)
    # print("Action Min:", a_min)
