import torch
from torch.utils.data import Dataset
import os
import pickle as pkl
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle
import torch.nn.functional as F
from einops import rearrange
import torchaudio.transforms as T
import torchaudio
from utils.meldataset import get_mel_spectrogram
import librosa

def createTrainDataset(root_dir='./data', batch_size=32, stride=64, sample_len=360, start=0, end=1):
    with open(os.path.join(root_dir, 'train.txt'), 'r') as file:
        file_names = [line.strip() for line in file]
    wavs = []
    for file_name in tqdm(file_names):
        wav_file_path = os.path.join(root_dir, 'wav', file_name+'.wav')
        wav_form, sr = librosa.load(wav_file_path, sr=24000, mono=True)
        wav_form = torch.FloatTensor(wav_form).unsqueeze(0)
        wav_form = get_mel_spectrogram(wav_form)
        wav_form = rearrange(wav_form, 'b c t -> b t c')

        total_length = wav_form.shape[1]
        for i in range(start, total_length+end-sample_len, stride):
            wavs.append(wav_form[:, i:i+sample_len, :])

    wavs = torch.cat(wavs, dim=0)
    print(wavs.shape)
    print('Train Dataset len: ', wavs.shape[0])
    train_dataloader = DataLoader(TrainDataset(wavs), batch_size=batch_size, shuffle=True)
    return train_dataloader


def createEvalDataset(root_dir='./data', batch_size=32, stride=100000, sample_len=768, start=0, end=1):
    with open(os.path.join(root_dir, 'test.txt'), 'r') as file:
        file_names = [line.strip() for line in file]
    wavs, file_names_plus = [], []
    for file_name in tqdm(file_names):
        wav_file_path = os.path.join(root_dir, 'wav', file_name+'.wav')
        wav_form, sr = librosa.load(wav_file_path, sr=24000, mono=True)
        wav_form = torch.FloatTensor(wav_form).unsqueeze(0)
        wav_form = get_mel_spectrogram(wav_form)
        wav_form = rearrange(wav_form, 'b c t -> b t c')
        
        total_length = wav_form.shape[1]
        for i in range(start, total_length+end-sample_len, stride):
            wavs.append(wav_form[:, i:i+sample_len, :])
            file_names_plus.append(file_name+f'_{str(i//stride)}')

    wavs = torch.cat(wavs, dim=0)
    print(wavs.shape)
    print('Eval Dataset len: ', wavs.shape[0])
    eval_dataloader = DataLoader(EvalDataset(wavs, file_names_plus), batch_size=batch_size, shuffle=False)
    return eval_dataloader


class TrainDataset(Dataset):
    def __init__(self, wavs):
        
        wavs_mean, wavs_std = torch.mean(wavs), torch.std(wavs)
        torch.save({
            'wavs_mean': wavs_mean,
        }, './Pretrained/mean.pt')
        torch.save({
            'wavs_std': wavs_std,
        }, './Pretrained/std.pt')
        print('Save MEAN and STD!')
        self.wavs = wavs

    def __len__(self):
        return len(self.wavs)
    
    def __getitem__(self, idx):
        wav = self.wavs[idx]
        return {'wav': wav}

class EvalDataset(Dataset):
    def __init__(self, wavs, file_names):
        
        self.wavs = wavs
        self.file_names = file_names
        

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        wav = self.wavs[idx]
        file_name = self.file_names[idx]
        return {'wav': wav, 'file_name': file_name}

