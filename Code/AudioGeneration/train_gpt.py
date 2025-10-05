import torch
import torch.optim as optim
from utils.metric_vae import Metric
# from utils.metric_vae import Metric
from utils.savefig import savefig
from utils.utils import seed_everything
from models.GPT.dataloader import createTrainDataset, createEvalDataset
from models.GPT.trainer import Trainer
from utils.adan import Adan
from datetime import datetime
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
from utils.load_model import load_model
import argparse


def train(model, device, train_loader, optimizer, epoch):
    losses = []
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        loss = model(data, device, epoch)
        loss['total'].backward()
        losses.append(loss['total'])
        optimizer.step()
    torch.cuda.empty_cache()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Train Epoch: {epoch} | Timestep: {current_time} | Loss: {float(sum(losses)/len(losses))}')


def validate(model, device, eval_loader, epoch, root_dir, exp_name):
    losses = []
    metric = Metric(root_dir)
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(eval_loader)):
            result, loss = model.inference(data, device)
            losses.append(loss['total'])
            metric.update(result)
            savefig(model, result, epoch, exp_name)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Eval Epoch: {epoch} | Timestep: {current_time} | Loss: {float(sum(losses)/len(losses))}')
    metric.result()

def main():

    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Training script')
    # 添加 root_dir 参数
    parser.add_argument('--root_dir', type=str, default='./data1', help='Root directory of the data')
    # 添加 exp_name 参数
    parser.add_argument('--exp_name', type=str, default='gpt', help='Experiment name')
    # 解析命令行参数
    args = parser.parse_args()

    seed_everything(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_bs, val_bs = 64, 32

    root_dir = args.root_dir
    exp_name = args.exp_name

    model_args = {'device': device}
    model = Trainer(**model_args).to(device)
    
    optimizer = Adan(model.parameters(), lr=4e-4, weight_decay=0.02)

    print('Loading Data')
    print('Construct Training Data')
    train_loader = createTrainDataset(root_dir=root_dir, batch_size=train_bs)
    print('Construct Eval Data')
    eval_loader = createEvalDataset(root_dir=root_dir, batch_size=val_bs)
    print('Start Training!')

    for epoch in range(1, 2001):
        train(model, device, train_loader, optimizer, epoch)
        if epoch % 200 == 0:
            validate(model, device, eval_loader, epoch, root_dir, exp_name)
        # validate(model, device, eval_loader, epoch, root_dir, exp_name)

if __name__ == '__main__':
    main()
