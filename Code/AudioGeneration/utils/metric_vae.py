import torch
import torch.nn as nn
import os
from smplx import SMPL
import torch
import numpy as np
import pickle as pkl
from scipy import linalg
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
from einops import rearrange
from tqdm import tqdm

class Metric():
    def __init__(self, root_dir='./data'):
        
        self.MAE_Loss = torch.nn.L1Loss()
        self.MSE_Loss = nn.MSELoss()

        self.wav_pred = []
        self.wav_gt = []

    def update(self, result):
        self.wav_pred.append(result['wav_pred'].detach().cpu())
        self.wav_gt.append(result['wav_gt'].detach().cpu())

    def result(self):
        wav_gt, wav_pred = torch.cat(self.wav_gt, dim=0), torch.cat(self.wav_pred, dim=0)
        print("MSE Loss: ", self.MSE_Loss(wav_gt, wav_pred))
        print("MAE Loss: ", self.MAE_Loss(wav_gt, wav_pred))

