import os
import glob
import torch
import librosa
import pickle
from tqdm import tqdm
from muq import MuQ
import torch.nn.functional as F

def upsample_features(features, original_fps=25, target_fps=30):
    """将特征张量从原始帧率上采样到目标帧率"""
    # 原始特征形状: [batch_size, time_steps, features]
    b, t, c = features.shape
    
    # 计算目标长度 (t * 6/5)
    target_t = int(t * target_fps / original_fps)  # 等效于t*6/5
    
    # 转换为适合插值的维度 [batch, channels, time]
    features = features.permute(0, 2, 1)
    
    # 线性插值上采样
    upsampled = F.interpolate(
        features,
        size=target_t,
        mode='linear',
        align_corners=False
    )
    
    # 恢复原始维度 [batch, time, channels]
    return upsampled.permute(0, 2, 1)

def extract_muq(a_folder, b_folder):
    os.makedirs(b_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    muq = MuQ.from_pretrained("./Pretrained/MuQ-large-msd-iter", local_files_only=True)
    muq = muq.to(device).eval()
    model = muq

    for wav_path in tqdm(glob.glob(os.path.join(a_folder, "*.wav"))):
        try:
            # 加载音频 (自动重采样到24kHz)
            wav, _ = librosa.load(wav_path, sr=24000)
            wav_tensor = torch.as_tensor(wav).unsqueeze(0).to(device)
            
            # 提取特征
            with torch.no_grad():
                output = model(wav_tensor, output_hidden_states=True)
            
            # 只取最后一层特征 [1, t, c]
            last_hidden = output.last_hidden_state
                
            # 上采样到30fps
            upsampled = upsample_features(last_hidden)[0].cpu().numpy()

            print(wav_path, upsampled.shape)
            
            # 构建保存路径
            pkl_path = os.path.join(
                b_folder,
                os.path.basename(wav_path).replace(".wav", ".pkl")
            )
            
            # 保存为字典 (保持CPU上的float32)
            with open(pkl_path, "wb") as f:
                pickle.dump(
                    {"music": upsampled},
                    f,
                )
                
        except Exception as e:
            print(f"处理失败: {wav_path} - {str(e)}")
            continue

if __name__ == "__main__":

    
    # 设置路径 (根据实际情况修改)
    input_folder = "/data3/yangkaixing/CustomDance/GPT/Genre-Control-Deep/data/FineDance/music"
    output_folder = "/data3/yangkaixing/CustomDance/GPT/Genre-Control-Deep/data/FineDance/muq"
    
    process_audio_folder(input_folder, output_folder)