# from Bio import SeqIO
# import os
#
# dir_path = '/Users/jerrychen/PycharmProjects/RNA-protein-recovered/datasets /clip'
#
# train_seq = []
# train_des = []
# test_seq = []
# test_des = []
#
# train_seq_list = []
# train_des_list = []
# test_seq_list = []
# test_des_list = []
#
# name = []
#
# train_seq_k_mer = []
# def get_dir(path):
#         dirlist = []
#         for file in os.listdir(path):
#             dirlist.append(file)
#         return dirlist
#
# filename_list = get_dir(dir_path)
# for i in range(len(filename_list)):
#     if (filename_list[i] == '.DS_Store'):
#         continue
#     name.append(filename_list[i])
#     train_path = dir_path + '/' + filename_list[i] + '/30000/training_sample_0/sequences.fa'
#     test_path = dir_path + '/' + filename_list[i] + '/30000/test_sample_0/sequences.fa'
#     with open(train_path, 'r') as handles:
#         for record in SeqIO.parse(handles, "fasta"):
#             train_seq_list.append(record.seq)
#             print(record.seq)
#             train_des_list.append(record.description)
#             print(record.description)
#         train_seq.append(train_seq_list)
#         train_des.append(train_des_list)
#         train_seq_list = []
#         train_des_list = []
#     with open(test_path, "r") as handles:
#         for record in SeqIO.parse(handles, 'fasta'):
#             test_seq_list.append(record.seq)
#             print(record.seq)
#             test_des_list.append(record.description)
#             print(record.description)
#         test_seq.append(test_seq_list)
#         test_des.append(test_des_list)
#         test_seq_list = []
#         test_des_list = []

import numpy as np
k_mer = 6
train_seq = np.load('data_preprocess/train_seq.npy', allow_pickle=True)
train_seq_k = []
for i in range(len(train_seq)):
    for j in range(len(train_seq[i])):
        for k in range(len(train_seq[i][j])-k_mer):
            train_seq_k.append(train_seq[i][j][k : k+k_mer])



train_des = np.load('data_preprocess/train_des.npy', allow_pickle=True)
train_label_k = []

for i in range(len(train_des)):
    for j in range(len(train_des[i])):
        if 'class:0' in train_des[i][j]:
            for a in range(len(train_seq[i][j])-k_mer):
                train_label_k.append('0')
        else:
            for a in range(len(train_seq[i][j])-k_mer):
                train_label_k.append('1')

test_seq = np.load('data_preprocess/test_seq.npy', allow_pickle=True)
test_seq_k = []
for i in range(len(test_seq)):
    for j in range(len(test_seq[i])):
        for k in range(len(test_seq[i][j])-k_mer):
            test_seq_k.append(test_seq[i][j][k : k_mer + k])



test_des = np.load('data_preprocess/test_des.npy', allow_pickle=True)
test_label_k = []

for i in range(len(test_des)):
    for j in range(len(test_des[i])):
        if 'class:0' in test_des[i][j]:
            for a in range(len(test_seq[i][j])-k_mer):
                test_label_k.append('0')
        else:
            for a in range(len(test_seq[i][j])-k_mer):
                test_label_k.append('1')

np.save('train_seq_repetitive_6.npy', train_seq_k)
np.save('train_label_repetitive_6.npy', train_label_k)
np.save('test_seq_repetitive_6.npy', test_seq_k)
np.save('test_label_repetitive_6.npy', test_label_k)