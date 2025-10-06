import librosa
import soundfile as sf

# Load audio
audio, sr = librosa.load("/Users/jerrychen/PycharmProjects/music/music_generation_platform/static/uploads/The Musical Offering, BWV 1079 - Five Canons on the Royal Theme.wav", sr=None)

# Resample to 16 kHz
resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)

# Save resampled audio
sf.write("/Users/jerrychen/PycharmProjects/music/music_generation_platform/static/The Musical Offering, BWV 1079 - Five Canons on the Royal Theme.wav", resampled_audio, 48000)
