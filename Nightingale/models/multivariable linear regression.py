from feature_selection import valence_features
from feature_selection import arousal_features
from feature_selection import valence
from sklearn.preprocessing import scale
from matplotlib.font_manager import FontProperties
from feature_selection import arousal
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_num, hidden_num),
            nn.ReLU(),
            nn.Linear(hidden_num, output_num),
            nn.ReLU()
        )

    def forward(self, input):
        return self.net(input)

ori_valence_label = valence[40000:-1]
net = Net(input_num=100, hidden_num=101, output_num=1).to(device)
valence_features, valence = torch.FloatTensor(scale(valence_features)).to(device), torch.unsqueeze(torch.FloatTensor(valence), dim=1).to(device)

train_valence_features, train_valence_label = valence_features[0:40000,:], valence[0:40000]
valid_valence_features, valid_valence_label = valence_features[40000:-1,:], valence[40000:-1]

epochs = 60
learning_rate = 0.001
batch_size = 1024
total_step = int(train_valence_features.shape[0] / batch_size)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
loss_func = torch.nn.MSELoss()

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

net.apply(weight_reset)

epoch_train_valence_loss_value = []
step_train_valence_loss_value = []
epoch_valid_valence_loss_value = []

for i in range(epochs):
    for step in range(total_step):
        xs = train_valence_features[step * batch_size:(step + 1) * batch_size, :]
        ys = train_valence_label[step * batch_size:(step + 1) * batch_size]
        prediction = net(xs)
        loss = loss_func(prediction, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_train_valence_loss_value.append(loss.detach().numpy())
    valid_loss = loss_func(net(valid_valence_features), valid_valence_label)
    epoch_valid_valence_loss_value.append(valid_loss.detach().numpy())
    epoch_train_valence_loss_value.append(np.mean(step_train_valence_loss_value))
    print('epoch={:3d}/{:3d}, train_loss={:.4f}, valid_loss={:.4f}'.format(i + 1,epochs,np.mean(step_train_valence_loss_value),valid_loss))

prediction = []

for i in range(valid_valence_features.shape[0]):
    prediction.append(net(valid_valence_features[i, :]).item())

print(r2_score(ori_valence_label, prediction))



