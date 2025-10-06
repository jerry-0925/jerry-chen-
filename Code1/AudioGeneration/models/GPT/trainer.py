import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from utils.features.geometric import geometric_features
from utils.features.kinetic import kinetic_features
import yaml
from types import SimpleNamespace
from einops import rearrange
from utils.utils import denormalize, normalize, root_postprocess, keypoint_from_smpl, rotation_6d_to_angle_axis
from utils.load_model import load_model
from models.GPT.model2 import GPT
from models.FSQ.trainer import SepFSQ as FSQ
# from models.CLS.classification import CLS
from smplx import SMPL
import pickle
import json
import time

class Trainer(nn.Module):
    def __init__(self, device=None):
        super().__init__()

        # self.cls_model = CLS(device)
        # self.cls_model = load_model(self.cls_model, 'CLS', 30)
        # self.cls_model.eval()

        self.fsq = FSQ(device)
        self.fsq = load_model(self.fsq, 'fsq', 2000)
        self.fsq.eval()
        
        self.gpt = GPT(device)

    def forward(self, data, device, epoch):
        self.gpt.train()
        wav = self.fsq.preprocess(data, device)
        label = data['label'].to(device).long()
        wav = self.fsq.wav_encode(wav)
        wav1, wav2 = wav[:, :, 0], wav[:, :, 1]
        loss = self.gpt(label, wav1, wav2)
        loss = {'total': loss}
        return loss

    
    def inference(self, data, device, length):
        self.gpt.eval()
        wav = self.fsq.preprocess(data, device)
        label = data['label'].to(device).long()
        wav_idx = self.fsq.wav_encode(wav)
        wav1, wav2 = wav_idx[:, :, 0], wav_idx[:, :, 1]
        wav_pred = self.gpt.inference(label, wav1, wav2, length)
        wav_decoded = self.fsq.wav_decode(wav_pred) 

        wav_gt = self.fsq.postprocess(wav)
        wav_pred = self.fsq.postprocess(wav_decoded)

        result = {
            'wav_gt': wav_gt, 'wav_pred': wav_pred,  \
            'file_name': data['file_name'], \
        }

        return result, {'total': 0}


    def demo(self, data, device):
        self.gpt.eval()
        music_librosa, smpl_trans_gt, smpl_poses_gt, smpl_root_vel_gt, smpl_root_init_gt, _, label = self.fsq.preprocess(data, device)
        music_librosa_d = music_librosa[:, ::8, :]
        start_time = time.time()
        print(f'music Length : {music_librosa_d.shape[1]/3.75:.4f} second')
        pose_up_index, pose_down_index = self.fsq.pose_encode(smpl_poses_gt, smpl_root_vel_gt)
        # pose_up_index, pose_down_index = pose_up_index[:, :, 0], pose_down_index[:, :, 0]
        pose_up_index_pred, pose_down_index_pred = self.gpt.inference(music_librosa_d, pose_up_index, pose_down_index, label)
        # pose_up_index_pred, pose_down_index_pred = rearrange(pose_up_index_pred, 'b t -> b t 1'), rearrange(pose_down_index_pred, 'b t -> b t 1')
        root_vel_decoded, pose_decoded = self.fsq.pose_decode(pose_up_index_pred, pose_down_index_pred) 
        smpl_trans_pred, smpl_root_vel_pred, smpl_poses_pred, keypoints_pred, _, smpl_poses_axis_pred = self.fsq.postprocess(pose_decoded, root_vel_decoded, smpl_root_init_gt)
        end_time = time.time()
        print(f"generation time: {end_time - start_time:.4f} 秒")
        for i in range(16):
            with open(os.path.join(data['root_dir'][i], 'keypoint', data['file_name'][i].replace('.wav', '.json')), 'w') as f:
                json.dump({'keypoints': keypoints_pred[i].cpu().detach().numpy().tolist()}, f, indent=4)     
            with open(os.path.join(data['root_dir'][i], 'smpl', data['file_name'][i].replace('.wav', '.pkl')), 'wb') as f:
                pickle.dump({'smpl_trans': smpl_trans_pred[i].cpu().detach().numpy(), 'smpl_poses': smpl_poses_axis_pred[i].cpu().detach().numpy()}, f)  

