from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F
import math
# from models.GPT.kan import KAN
from mamba_ssm import Mamba
from mamba_ssm import Mamba2
import random


class CE_Loss(nn.Module):
    def __init__(self, ):
        super(CE_Loss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, wav_pred, wav_tgt):
        wav_pred= rearrange(wav_pred, 'b t c -> (b t) c').float()
        wav_tgt = rearrange(wav_tgt, 'b t -> (b t)').long()
        loss = self.cls_loss(wav_pred, wav_tgt)
        return loss


class GPT(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        self.dance_decoder = Dance_Decoder()
        self.cls_loss = CE_Loss()
        self.window_size, self.train_size = 30, 44

    def forward(self, label, wav1, wav2):
        wav1_src, wav1_tgt = wav1[:, :-1], wav1[:, 1:]
        wav2_src, wav2_tgt = wav2[:, :-1], wav2[:, 1:]
        b, t = wav1_src.shape
        label = label.repeat(1, t)
        wav1_pred, wav2_pred = self.dance_decoder(label, wav1_src, wav2_src)  
        loss = self.cls_loss(wav1_pred, wav1_tgt) + self.cls_loss(wav2_pred, wav2_tgt)
        return loss

    def inference(self, label, wav1, wav2, length=None):
        b, t = wav1.shape
        if length is not None:
            t = length * 90 // 8
        labels = label.repeat(1, t)
        label = labels[:, :1]
        wav1_index, wav1_indexs = wav1[:, :1], wav1[:, :1]
        wav2_index, wav2_indexs = wav2[:, :1], wav2[:, :1]
        for i in range(1, t):
            wav1_preds, wav2_preds = self.dance_decoder(label, wav1_index, wav2_index)
            _, wav1_pred = torch.max(wav1_preds[:, -1:, :], dim=-1)
            _, wav2_pred = torch.max(wav2_preds[:, -1:, :], dim=-1)
            wav1_indexs = torch.cat([wav1_indexs, wav1_pred], dim=1)
            wav2_indexs = torch.cat([wav2_indexs, wav2_pred], dim=1)
            if i < self.window_size:
                label = labels[:, :i+1]
                wav1_index = wav1_indexs
                wav2_index = wav2_indexs         
            else:
                label = labels[:, i-self.window_size+1:i+1]
                wav1_index = wav1_indexs[:, i-self.window_size+1:i+1]
                wav2_index = wav2_indexs[:, i-self.window_size+1:i+1]
        wav1_indexs, wav2_indexs = rearrange(wav1_indexs, 'b t -> b t 1'), rearrange(wav2_indexs, 'b t -> b t 1')
        wav_pred = torch.cat((wav1_indexs, wav2_indexs), dim=2)
        return wav_pred


class Dance_Decoder(nn.Module):
    def __init__(self, input_dim=512, output_dim=54, hidden_dim=512, num_layers=8, nhead=8, levels=None):
        super().__init__()
        self.layer = num_layers
        self.decoder_layers = nn.ModuleList([DecoderLayer(hidden_dim, nhead) for _ in range(num_layers)])
        self.label_emb = nn.Embedding(5, hidden_dim)
        self.wav1_emb = nn.Embedding(1000, hidden_dim)
        self.wav2_emb = nn.Embedding(1000, hidden_dim)
        self.wav1_output = nn.Sequential(
            nn.Linear(hidden_dim, 768),
            nn.Linear(768, 1000),
        )
        self.wav2_output = nn.Sequential(
            nn.Linear(hidden_dim, 768),
            nn.Linear(768, 1000),
        )

    def forward(self, label, wav1_src, wav2_src):
        b, t = wav1_src.shape
        label_feat = self.label_emb(label)
        wav1_src_feat = self.wav1_emb(wav1_src)
        wav2_src_feat = self.wav2_emb(wav2_src)
        feat_src = torch.cat([label_feat, wav1_src_feat, wav2_src_feat], dim=1)
        for i in range(self.layer):
            feat_src = self.decoder_layers[i](feat_src)
        wav1_pred, wav2_pred = feat_src[:, t:2*t, :], feat_src[:, 2*t:, :]
        wav1_pred, wav2_pred = self.wav1_output(wav1_pred), self.wav2_output(wav2_pred)
        return wav1_pred, wav2_pred


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        self.global_swab = SWABlock()

    def forward(self, feat_src):
        feat_src = self.global_swab(feat_src)
        return feat_src

class SWABlock(nn.Module):
    def __init__(self, n_embd=512, pdrop=0.4):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
        self.attn = SWACrossConditionalSelfAttention()
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x    

class LayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x):
        b, T, c = x.shape
        t = T // 3
        x_feat = torch.zeros_like(x).to(x.device)  
        x_feat[:, :t, :] += self.ln1(x[:, :t, :])
        x_feat[:, t:2*t, :] += self.ln2(x[:, t:2*t, :])
        x_feat[:, 2*t:, :] += self.ln3(x[:, 2*t:, :])
        return x_feat   

class SWACrossConditionalSelfAttention(nn.Module):
    def __init__(self, n_embd=512, pdrop=0.4, block_size=240, n_head=8):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(pdrop)
        self.resid_drop = nn.Dropout(pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.register_buffer("mask", self.get_mask())

    def get_mask(self, ):
        window_size, train_size = 30, 44
        mask = torch.triu(torch.ones((train_size, train_size), dtype=torch.bool), 1)
        for i in range(window_size, train_size):
            mask[i, :i-window_size+1] = True
        mask = mask.view(1, 1, 44, 44)
        return mask

    def forward(self, x):
        B, T, C = x.size()  # T = 2*t (music up down)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        t = T // 3
        mask = self.mask[:, :, :t, :t].repeat(1, 1, 3, 3)
        att = att.masked_fill(mask==1, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_drop(self.proj(y))
        return y
