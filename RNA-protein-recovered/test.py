

# dir_path = '/Users/jerrychen/PycharmProjects/RNA-protein-recovered/datasets /clip'
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
#     train_path = dir_path + '/' + filename_list[i] + '/30000/training_sample_0/sequences.fa'
#     test_path = dir_path + '/' + filename_list[i] + '/30000/test_sample_0/sequences.fa'
#     with open(train_path, 'r') as handles:
#         for record in SeqIO.parse(handles, "fasta"):
#             train_seq.append(record.seq)
#             print(record.seq)
#             train_des.append(record.description)
#             print(record.description)
#     with open(test_path, "r") as handles:
#         for record in SeqIO.parse(handles, 'fasta'):
#             test_seq.append(record.seq)
#             print(record.seq)
#             test_des.append(record.description)
#             print(record.description)


# train_seq = np.load('train_seq.npy', allow_pickle=True)
# test_seq = np.load('test_seq.npy', allow_pickle=True)
#
# train_single = []
# test_single = []
# train_dict = {}

# for i in range(len(train_seq)):
#     for j in range(len(train_seq[i])):
#         for k in range(len(train_seq[i][j])):
#             train_single.append(train_seq[i][j][k])
#



# for i in range(len(test_seq)):
#     for j in range(len(test_seq[i])):
#         for k in range(len(test_seq[i][j])):
#             test_single.append(test_seq[i][j][k])
#
# np.save("train_single.npy", train_single)
# np.save("test_single.npy", test_single)




import numpy as np
import re

# def string_to_array(my_string):
#     my_string = my_string.lower()
#     my_string = re.sub('[^acgt]', 'z', my_string)
#     my_array = np.array(list(my_string))
#     return my_array
#
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# label_encoder.fit(np.array(['a','c','g','t','z']))
#
# def ordinal_encoder(my_array):
#     integer_encoded = label_encoder.transform(my_array)
#     float_encoded = integer_encoded.astype(float)
#     float_encoded[float_encoded == 0] = 0.261 # A
#     float_encoded[float_encoded == 1] = 0.224 # C
#     float_encoded[float_encoded == 2] = 0.224 # G
#     float_encoded[float_encoded == 3] = 0.289 # T
#     float_encoded[float_encoded == 4] = 0.002 # anything else, z
#     return float_encoded
#
#
# train_seq_20 = np.load('train_seq_20.npy', allow_pickle=True)
# test_seq_20 = np.load('test_seq_20.npy', allow_pickle=True)
#
#
# string = ''
# for i in range(len(train_seq_20)):
#     for j in range(len(train_seq_20[i])):
#         string += train_seq_20[i][j]
#
#
# a = ordinal_encoder(string_to_array(string))
# b = ordinal_encoder(string_to_array(test_seq_20))
#
# print(a)
# np.save("train_encode.npy", a)
# np.save("test_encode.npy", b)

a = np.load('data_preprocess/train_encode_list_20.npy')
b = np.load('data_preprocess/test_encode_list_20.npy')
print(a)
print(b)

