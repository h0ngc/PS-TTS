from pydub import AudioSegment
import os
import librosa
import numpy as np
def find_silence_intervals(mask, hop_length=256, sr=22050):
    intervals = []
    current_start = None
    for i, m in enumerate(mask):
        if m and current_start is None:
            current_start = i
        elif not m and current_start is not None:
            intervals.append((current_start, i))
            current_start = None
    if current_start is not None:
        intervals.append((current_start, len(mask)))

    # 0.4초 이상 지속되는 구간 찾기
    min_length = int(0.3 * sr / hop_length)
    long_intervals = [(start, end) for start, end in intervals if (end - start) >= min_length]
    silence_length = sum(end - start for start, end in long_intervals)
    return long_intervals,silence_length

# def adjust_volumes(waveform):
def fsilence(y, sr):
    # RMS 에너지 계산
    hop_length = 256
    rms = librosa.feature.rms(y=y, hop_length=hop_length).flatten()
    # 임계값 설정
    threshold = 0.001
    # Silence 부분 탐지
    silence_mask = rms < threshold
    # 전체 프레임 수 계산
    total_frames = len(rms)
    # Silence가 아닌 프레임 수 계산
    silence_intervals,silence_length = find_silence_intervals(silence_mask)
    non_silence_frames = total_frames-silence_length

    return total_frames, silence_intervals,non_silence_frames