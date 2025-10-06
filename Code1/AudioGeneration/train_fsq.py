import torch
import torch.optim as optim
from utils.metric_vae import Metric
from utils.savefig import savefig
from utils.utils import seed_everything
from utils.adan import Adan
from models.FSQ.dataloader import createTrainDataset, createEvalDataset
from models.FSQ.trainer import SepFSQ as FSQ
from datetime import datetime
from tqdm import tqdm

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_idx, data in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        _, loss = model(data, device)
        loss['total'].backward()
        losses.append(loss['total'])
        optimizer.step()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Train Epoch: {epoch} | Timestep: {current_time} | Loss: {float(sum(losses)/len(losses))}')


def validate(model, device, eval_loader, epoch, root_dir, exp_name):
    model.eval()
    metric = Metric(root_dir)
    losses = []
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(eval_loader)):
            result, loss = model.inference(data, device)
            losses.append(loss['total'])
            metric.update(result)
            savefig(model, result, epoch, exp_name)
            
    # save_model(model, epoch, exp_name)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Eval Epoch: {epoch} | Timestep: {current_time} | Loss: {float(sum(losses)/len(losses))}')
    metric.result()


def main():

    seed_everything(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128

    root_dir = './data1'
    exp_name = 'fsq'

    print('Loading Data')
    print('Construct Training Data')
    train_loader = createTrainDataset(root_dir=root_dir, batch_size=batch_size)
    print('Construct Eval Data')
    eval_loader = createEvalDataset(root_dir=root_dir, batch_size=batch_size)

    model_args = {'device': device}
    model = FSQ(**model_args).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=3e-5, betas=[0.5, 0.999])
    optimizer = Adan(model.parameters(), lr=4e-4, weight_decay=0.02)

    print('Start Training!')
    for epoch in range(1, 2001):
        train(model, device, train_loader, optimizer, epoch)
        if epoch % 200 == 0:
            validate(model, device, eval_loader, epoch, root_dir, exp_name)

if __name__ == '__main__':
    main()
