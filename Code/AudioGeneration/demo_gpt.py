import torch
import torch.optim as optim
from utils.metric import Metric
from utils.savefig import savefig
from utils.utils import seed_everything
from models.GPT.dataloader import createDemoDataset
from models.GPT.trainer import Trainer
from datetime import datetime
from tqdm import tqdm
from utils.load_model import load_model

def validate(model, device, eval_loader, epoch, file_name, exp_name, length):
    model.eval()
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(eval_loader)):
            result, loss = model.inference(data, device, length)
            wav_pred = result['wav_pred'].detach().cpu()
            torch.save(wav_pred, file_name)

def main():

    seed_everything(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    exp_name = 'gpt'
    epoch = 2000
    label = 2
    file_name ='./demo/pt/2.pt' # 存储在哪个位置
    length = 40 # 秒为单位

    print('Loading Data')
    print('Construct Eval Data')
    eval_loader = createDemoDataset(label, length, file_name)
    print('Start Training!')

    model_args = {'device': device}
    model = Trainer(**model_args).to(device)
    model = load_model(model, exp_name, epoch)
        
    validate(model, device, eval_loader, epoch, file_name, exp_name, length)

if __name__ == '__main__':
    main()
