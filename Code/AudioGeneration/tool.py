from pydub import AudioSegment

# 输入音频文件路径
input_path = "/data3/yangkaixing/CustomDance/YF/Code/AudioGeneration/data1/wav/dstestsoftcream1.wav"
# 输出音频文件路径
output_path = "/data3/yangkaixing/CustomDance/YF/Code/AudioGeneration/Pretrained/init_wav/2.wav"

# 裁剪区间（单位：秒）
start_time = 2   # a 秒
end_time = 5    # b 秒

# 读取音频
audio = AudioSegment.from_file(input_path)

# 裁剪并导出
cropped = audio[start_time * 1000 : end_time * 1000]  # pydub 以毫秒为单位
cropped.export(output_path, format="wav")

print(f"裁剪完成：{output_path}")