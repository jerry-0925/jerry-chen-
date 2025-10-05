import numpy as np
import torch
import os

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

class RNADataset(Dataset):
    def __init__(self, data_dir, encode_filename_path,label_filename_path):
        self.data_dir = data_dir
        self.encode_filename_path = encode_filename_path
        self.label_filename_path = label_filename_path
        # self.encode_file_name = "train_encode_list_20.npy"
        # self.label_file_name = "train_label_20.npy"
        self.encode = np.load(os.path.join(self.data_dir,self.encode_filename_path))
        self.label = np.load(os.path.join(self.data_dir, self.label_filename_path))
        self.length = len(self.encode)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rna_seq = self.encode[index]
        rna_label = self.label[index]
        rna_seq = torch.Tensor(rna_seq)
        rna_label = torch.Tensor(rna_label)
        return rna_seq,rna_label

dataset = RNADataset('data_preprocess', 'train_encode_list_20.npy','train_label_20.npy')
dataloader = DataLoader(dataset, batch_size=1)



