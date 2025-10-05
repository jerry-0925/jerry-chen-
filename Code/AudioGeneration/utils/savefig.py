import torch
import os
import torch
import pickle
import json
from utils.utils import root_postprocess, keypoint_from_smpl, rotation_6d_to_angle_axis
import torchaudio
import torchaudio.transforms as T

output_dir = './output/'
os.makedirs(output_dir, exist_ok=True)
def savefig(model, result, epoch, exp_name):

    wav_gt, wav_pred, file_name = result['wav_gt'].detach().cpu(), result['wav_pred'].detach().cpu(), result['file_name']
    output_gt_dir = os.path.join(output_dir, exp_name, str(epoch), 'wav_gt')
    output_pred_dir = os.path.join(output_dir, exp_name, str(epoch), 'wav_pred')
    os.makedirs(output_gt_dir, exist_ok=True)
    os.makedirs(output_pred_dir, exist_ok=True)


    for i in range(len(file_name)):
        # output_gt_file = os.path.join(output_gt_dir, file_name[i]+'.wav')
        # torchaudio.save(output_gt_file, wav_gt[i:i+1], 16000)
        # output_pred_file = os.path.join(output_pred_dir, file_name[i]+'.wav')
        # torchaudio.save(output_pred_file, wav_pred[i:i+1], 16000)
        output_gt_file = os.path.join(output_gt_dir, file_name[i]+'.pt')
        torch.save(wav_gt[i:i+1], output_gt_file)
        output_pred_file = os.path.join(output_pred_dir, file_name[i]+'.pt')
        torch.save(wav_pred[i:i+1], output_pred_file)


    output_model_file = os.path.join(output_dir, exp_name, str(epoch), 'Model')
    os.makedirs(output_model_file, exist_ok=True)
    torch.save({'model': model.state_dict()}, os.path.join(output_model_file, 'model.pth'))




def save_model(model, epoch, exp_name):
    output_model_file = os.path.join(output_dir, exp_name, str(epoch), 'Model')
    os.makedirs(output_model_file, exist_ok=True)
    torch.save({'model': model.state_dict()}, os.path.join(output_model_file, 'model.pth'))
