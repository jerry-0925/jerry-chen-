import numpy as np
import torch.nn as nn
from .fsq import Conv1DEncoder, Conv1DQuantizer, Conv1DDecoder
import yaml
from types import SimpleNamespace
import torch
from einops import rearrange
from utils.utils import normalize, denormalize
import torch as t
import torchaudio.transforms as T
import torchaudio
import sys
import os

class SepFSQ(nn.Module):
    def __init__(self, device):
        super().__init__()
        with open('./config/fsq.yaml', 'r') as f:
            hps = yaml.safe_load(f)
            hps = SimpleNamespace(**hps)
        self.hps = SimpleNamespace(**hps.solo)

        self.Encoder = Conv1DEncoder(self.hps, 100)
        self.Quantizer = Conv1DQuantizer(self.hps)
        self.Decoder = Conv1DDecoder(self.hps, 100)

        self.audio_loss = nn.MSELoss()

        mean, std = torch.load('./Pretrained/mean.pt'), torch.load('./Pretrained/std.pt')
        self.wav_mean = mean['wavs_mean'].to(device).float()
        self.wav_std = std['wavs_std'].to(device).float()    


    def forward(self, data, device):
        wav = self.preprocess(data, device)
        wav_encoded = self.Encoder(wav)
        _, wav_quantised = self.Quantizer(wav_encoded)
        wav_decoded = self.Decoder(wav_quantised)
        total_loss = self.audio_loss(wav_decoded, wav)
        loss = {'total': total_loss}
        return None, loss


    def inference(self, data, device):
        wav = self.preprocess(data, device)
        # wav_index = self.wav_encode(wav)
        # wav_decoded = self.wav_decode(wav_index)

        wav_encoded = self.Encoder(wav)
        _, wav_quantised = self.Quantizer(wav_encoded)
        wav_decoded = self.Decoder(wav_quantised)
    
        wav_gt = self.postprocess(wav)
        wav_pred = self.postprocess(wav_decoded)

        result = {
            'wav_gt': wav_gt, 'wav_pred': wav_pred,  \
            'file_name': data['file_name'], \
        }

        return result, {'total': 0}


    def wav_encode(self, wav):
        b, t, c = wav.size()
        wav_encoded = self.Encoder(wav)
        wav_index, wav_quantised = self.Quantizer(wav_encoded)
        return wav_index[0]

    def wav_decode(self, wav_idx):
        b, t, _ = wav_idx.shape
        t = 8 * t
        wav_quantised = self.Quantizer.get_feature_from_index(wav_idx)
        wav_decoded = self.Decoder(wav_quantised)
        return wav_decoded


    def preprocess(self, data, device):
        wav = data['wav'].to(device).float()
        wav = normalize(wav, self.wav_mean, self.wav_std)
        return wav

    def postprocess(self, wav):
        wav = denormalize(wav, self.wav_mean, self.wav_std)
        b, t, c = wav.shape
        wav = rearrange(wav, 'b t c -> b c t')
        return wav