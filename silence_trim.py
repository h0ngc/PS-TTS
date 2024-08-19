import librosa
import numpy as np
from pydub import AudioSegment

def load_intervals_from_txt(file_path):
    intervals = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                start_time = float(parts[1])
                end_time = float(parts[2])
                senten = parts[3]
                intervals.append((start_time, end_time, senten))
    return intervals
def find_speech_intervals(audio_path, intervals, threshold=0.001, hop_length=256):
    y, sr = librosa.load(audio_path, sr=None)
    speech_intervals = []

    for interval in intervals:
        start_sample = int(interval[0] * sr)
        end_sample = int(interval[1] * sr)
        audio_clip = y[start_sample:end_sample]
        rms = librosa.feature.rms(y=audio_clip, hop_length=hop_length).flatten()
        above_threshold_indices = np.where(rms > threshold)[0]
        if len(above_threshold_indices) > 0:

            speech_start_idx = above_threshold_indices[0]
            speech_end_idx = above_threshold_indices[-1]
            speech_start_sec = start_sample / sr + (speech_start_idx * hop_length) / sr
            speech_end_sec = start_sample / sr + (speech_end_idx * hop_length) / sr

            speech_intervals.append((speech_start_sec, speech_end_sec, interval[2]))
        else:
            speech_intervals.append((interval[0], interval[1], interval[2]))

    return speech_intervals
