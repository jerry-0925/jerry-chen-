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
    wavs, labels = [], []
    for file_name in tqdm(file_names):
        label_file_path = os.path.join(root_dir, 'label', file_name+'.txt')
        with open(label_file_path, 'r') as f:
            label = int(f.readline().strip())
            label = torch.tensor([label], dtype=torch.int)
            label = rearrange(label, '1 -> 1 1')
        wav_file_path = os.path.join(root_dir, 'wav', file_name+'.wav')
        wav_form, sr = librosa.load(wav_file_path, sr=24000, mono=True)
        wav_form = torch.FloatTensor(wav_form).unsqueeze(0)
        wav_form = get_mel_spectrogram(wav_form)
        wav_form = rearrange(wav_form, 'b c t -> b t c')

        total_length = wav_form.shape[1]
        for i in range(start, total_length+end-sample_len, stride):
            wavs.append(wav_form[:, i:i+sample_len, :])
            labels.append(label)

    wavs, labels = torch.cat(wavs, dim=0), torch.cat(labels, dim=0)
    print(wavs.shape, labels.shape)
    print('Train Dataset len: ', wavs.shape[0])
    train_dataloader = DataLoader(TrainDataset(wavs, labels), batch_size=batch_size, shuffle=True)
    return train_dataloader


def createEvalDataset(root_dir='./data', batch_size=32, stride=100000, sample_len=768, start=0, end=1):
    with open(os.path.join(root_dir, 'test.txt'), 'r') as file:
        file_names = [line.strip() for line in file]
    wavs, labels, file_names_plus = [], [], []
    for file_name in tqdm(file_names):
        label_file_path = os.path.join(root_dir, 'label', file_name+'.txt')
        with open(label_file_path, 'r') as f:
            label = int(f.readline().strip())
            label = torch.tensor([label], dtype=torch.int)
            label = rearrange(label, '1 -> 1 1')
        wav_file_path = os.path.join(root_dir, 'wav', file_name+'.wav')
        wav_form, sr = librosa.load(wav_file_path, sr=24000, mono=True)
        wav_form = torch.FloatTensor(wav_form).unsqueeze(0)
        wav_form = get_mel_spectrogram(wav_form)
        wav_form = rearrange(wav_form, 'b c t -> b t c')
        
        total_length = wav_form.shape[1]
        for i in range(start, total_length+end-sample_len, stride):
            wavs.append(wav_form[:, i:i+sample_len, :])
            labels.append(label)
            file_names_plus.append(file_name+f'_{str(i//stride)}')

    wavs, labels = torch.cat(wavs, dim=0), torch.cat(labels, dim=0)
    print(wavs.shape, labels.shape)
    print('Eval Dataset len: ', wavs.shape[0])
    eval_dataloader = DataLoader(EvalDataset(wavs, labels, file_names_plus), batch_size=batch_size, shuffle=False)
    return eval_dataloader


def createDemoDataset(label=0, length=40, file_name='./demo/1.pkl'):

    wavs, labels, file_names = [], [], []

    wav_file_path = f'./Pretrained/init_wav/{str(label)}.wav'
    wav_form, sr = librosa.load(wav_file_path, sr=24000, mono=True)
    wav_form = torch.FloatTensor(wav_form).unsqueeze(0)
    wav_form = get_mel_spectrogram(wav_form)
    wav_form = rearrange(wav_form, 'b c t -> b t c')
    wavs.append(wav_form)

    label = torch.tensor([label], dtype=torch.long)
    label = rearrange(label, '1 -> 1 1')
    labels.append(label)

    file_names.append(file_name)

    wavs, labels = torch.cat(wavs, dim=0), torch.cat(labels, dim=0)
    print(wavs.shape, labels.shape)
    print('Eval Dataset len: ', wavs.shape[0])
    eval_dataloader = DataLoader(EvalDataset(wavs, labels, file_names), batch_size=1, shuffle=False)
    return eval_dataloader

class TrainDataset(Dataset):
    def __init__(self, wavs, labels):
        
        self.wavs = wavs
        self.labels = labels

    def __len__(self):
        return len(self.wavs)
    
    def __getitem__(self, idx):
        wav = self.wavs[idx]
        label = self.labels[idx]
        return {'wav': wav, 'label': label}

class EvalDataset(Dataset):
    def __init__(self, wavs, labels, file_names):
        self.wavs = wavs
        self.labels = labels
        self.file_names = file_names
        
    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        wav = self.wavs[idx]
        label = self.labels[idx]
        file_name = self.file_names[idx]
        return {'wav': wav, 'label': label, 'file_name': file_name}

