import librosa
import numpy as np

# wav 파일 불러오기
signal, sr = librosa.load("duration_sample/casino.wav", sr=None)

# 에너지 계산
energy = librosa.feature.rms(signal)

# 특정 energy 값 이하인 부분 찾기
silence_frames = np.where(energy < threshold)[0]

# silence 구간 길이 계산
silence_lengths = np.diff(silence_frames)

# silence 구간 출력
for start, length in zip(silence_frames, silence_lengths):
    print(f"시작 시간: {start / sr:.2f}초, 길이: {length / sr:.2f}초")