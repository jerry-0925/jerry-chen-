import os
import torch
import soundfile as sf
import bigvgan
from tqdm import tqdm

device = 'cuda'

# 加载模型
model = bigvgan.BigVGAN.from_pretrained('.', use_cuda_kernel=False)
model.remove_weight_norm()
model = model.eval().to(device)

# 输入输出目录
input_dir = '/data3/yangkaixing/CustomDance/YF/Code/AudioGeneration/demo/pt'
output_dir = '/data3/yangkaixing/CustomDance/YF/Code/AudioGeneration/demo/wav'
os.makedirs(output_dir, exist_ok=True)

# 遍历所有.pt文件
for fname in tqdm(os.listdir(input_dir)):
    if fname.endswith('.pt'):
        mel_path = os.path.join(input_dir, fname)
        mel = torch.load(mel_path).to(device)

        with torch.inference_mode():
            wav_gen = model(mel)

        wav_gen_float = wav_gen.squeeze().cpu()
        wav_out_path = os.path.join(output_dir, fname.replace('.pt', '.wav'))
        sf.write(wav_out_path, wav_gen_float.numpy(), model.h.sampling_rate, subtype='PCM_16')

        print(f"Saved: {wav_out_path}")
