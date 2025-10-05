import torch.nn as nn
import torch
from load_dataset import RNADataset
from torch.utils.data.dataloader import DataLoader
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[-1, :, :])
        return out

input_size = 20  # 输入特征数
hidden_size = 20  # 隐藏层特征数
num_layers = 2  # LSTM层数
output_size = 1  # 输出类别数
batch_size = 1  # 批大小


# dataset = RNADataset('data_preprocess', 'train_encode_list_20.npy','train_label_20.npy')
# dataloader = DataLoader(dataset, batch_size=1)

model = LSTM(input_size, hidden_size, num_layers, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 开始训练
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(dataset)
    loss = criterion(outputs, dataloader)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# with torch.no_grad():
#     test_x = torch.randn(sequence_length, batch_size, input_size)
#     outputs = model(test_x)
#     _, predicted = torch.max(outputs.data, 1)
#     print(predicted)

from matplotlib import pyplot as plt

