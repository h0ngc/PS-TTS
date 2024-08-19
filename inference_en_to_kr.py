import logging
import os
import warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('numba.core.byteflow').setLevel(logging.CRITICAL)
logging.getLogger('numba').setLevel(logging.CRITICAL)
logging.getLogger('phonemizer').setLevel(logging.CRITICAL)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.CRITICAL)
logging.getLogger("tokenizers").setLevel(logging.CRITICAL)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os
from pathlib import Path
import pdb
import subprocess
import openai
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from vits_inference_ddpkr import Inference
from check_silence import fsilence
from silence_trim import load_intervals_from_txt, find_speech_intervals
import librosa
import numpy as np
import soundfile as sf
import torch
import resampy
import re
import json
import time




def replace_non_one_with_a(kr_list):
    return ['a' if x != 1 else 1 for x in kr_list]

def find_vowel_ranges(kr_list, vowel):
    ranges = []
    start = None

    for i, value in enumerate(kr_list):
        if value == vowel and start is None:
            start = i
        elif value != vowel and start is not None:
            ranges.append((start, i - 1))
            start = None

    if start is not None:
        ranges.append((start, len(kr_list) - 1))

    return ranges

def write_ranges_to_file(ranges, filename):
    with open(filename, 'w') as file:
        for start, end in ranges:
            file.write(f"{start} to {end}\n")


def main():
    parser = argparse.ArgumentParser(description="Run inference for Korean to English")

    parser.add_argument('--lang', type=str, default='korean', help="Language to use, e.g., 'korean'")
    parser.add_argument('--src_speech', type=str, required=True, help="Path to the source speech .wav file")
    parser.add_argument('--src_bgm', type=str, required=True, help="Path to the source BGM .wav file")
    parser.add_argument('--src_text', type=str, required=True, help="Path to the source text file")
    parser.add_argument('--trg_speech', type=str, required=True, help="Path to the target speech file")

    args = parser.parse_args()

    # Use the parsed arguments
    print(f"Source Speech: {args.src_speech}")
    print(f"Source BGM: {args.src_bgm}")
    print(f"Source Text: {args.src_text}")

    in_fpath = args.src_speech
    in_ffpath = args.src_bgm
    txt_fpath = args.src_text
    out_fpath = args.trg_speech
    ckpt = 'ckpts/baseline.pth'
    inference = Inference(ckpt)
    model = SentenceTransformer('sentence-transformers/LaBSE')
    script_path1 = './align_trans_kr.sh'

    try:
        subprocess.run([script_path1, txt_fpath], check=True)
    except subprocess.CalledProcessError as e:
        print(json.dumps({'error': f'{script_path1} script execution failed'}))
        return
    print('translation end')

    txt_en_fpath = f'./korean_trans.txt'
    #openAI config 파일 load
    with open('config.json') as f:
        config = json.load(f)
    openai.api_key = config['OPENAI_API_KEY']
    engine = config['OPENAI_ENGINE']


    with open(txt_fpath, 'r', encoding = 'utf-8') as fid: # source english text
        txts = fid.read().splitlines()
    with open(txt_en_fpath, 'r', encoding = 'utf-8') as line: # translated korean text
        lines = line.read().splitlines()


    # vocal load 
    vocal_wav, sample_rate = librosa.load(in_fpath, sr=22050, mono=True) # sr --> 22050 
    vocal_rms = np.sqrt(np.mean(vocal_wav**2))

    if os.path.exists(in_ffpath):
        bgm_wav, sample_bgm = librosa.load(in_ffpath,mono=True)
    else:
        bgm_wav = np.zeros(vocal_wav.shape / 22050 * 48000)

    new_vocal_audio = np.zeros(vocal_wav.shape)
    new_audio = np.zeros(bgm_wav.shape)

    prev_offset = 0
    start_time = time.time()  
    i=0
    for txt,line in zip(txts,lines):
        try :
            idx, onset, offset, kr_original = txt.split('\t')
            en_idx, en_onset, en_offset, en_original = line.split('\t')
        except Exception as e:
            print(e)
            continue
        word_count = len(kr_original.split()) 
        onset1 = int(float(onset)*sample_rate)
        offset1 = int(float(offset)*sample_rate)
        wav_seg = vocal_wav[onset1:offset1]
        sound_power = 0
        total_frame, silence_interval, non_silence_frames = fsilence(wav_seg, 22050)
        if total_frame==non_silence_frames :
            origin_frame = total_frame
        else :
            origin_frame = non_silence_frames
        new_length_scale, current_frames, spk_emb = inference.infer_origin(wav_seg, en_original, sample_rate, 0, origin_frame)
        if word_count <=1 : 
            if new_length_scale > 2.0 :
                new_length_scale = 1.0
            wav_out = inference.infer(wav_seg, kr_original, sample_rate, 1, spk_emb,new_length_scale)
            remaining_length = len(new_vocal_audio) - onset1
            actual_length = min(remaining_length, len(wav_out))
            adjusted_wav_out = wav_out[:actual_length]
            new_vocal_audio[onset1:onset1+actual_length] += adjusted_wav_out
            prev_offset = offset1
            i +=1
        else :
            new_translate ,w_sum= inference.infer_t(wav_seg, kr_original, sample_rate, 1, new_length_scale, origin_frame,model, spk_emb,en_original)
            response = openai.Completion.create(
                                    engine=engine,
                                    prompt=f"이 문장과 길이가 같고 의미가 같으며, 구어체로 된 문장을 60개를 만드세요. 또한, 이름이나 고유명사를 바꾸지 마세요. 각 문장을 숫자나 다른 기호 없이 작성하세요.: '{new_translate}'",
                                    max_tokens=1400,
                                    n=1,
                                    temperature=1,
                                    top_p=1
                                )
            response_text = response.choices[0].text.strip()
            cleaned_text = re.sub(r'\d+\.\s', '', response_text)
            sentences = cleaned_text.strip().split('\n')
            sentences.append(new_translate)
            en_list = inference.infer_en(wav_seg, en_original, sample_rate, 0, new_length_scale, spk_emb)
            score_total = []
            new_sentence = []
            rep = 0 
            for sentence in sentences :
                if len(sentence) == 0 :
                    continue
                print(sentence)
                score,path =  inference.infer_vowel(wav_seg, sentence, sample_rate,1, new_length_scale, spk_emb, en_list, rep, i, origin_frame)
                if score ==0 :
                    continue
                score_total.append(score)
                new_sentence.append(sentence)
                rep +=1
            if len(score_total) == 0 :
                score_total.append((0,origin_frame,w_sum, (origin_frame/w_sum)*new_length_scale))
                new_sentence.append(new_translate)
            sorted_by_dtw = sorted(score_total, key=lambda x: x[1])[:1]
            filtered_frames = [x for x in sorted_by_dtw if abs(x[2] - origin_frame) < 2 * origin_frame] 
            if filtered_frames:
                closest_frame = min(filtered_frames, key=lambda x: abs(x[2] - origin_frame))
            else :
                closest_frame = min(sorted_by_dtw, key=lambda x: abs(x[2] - origin_frame))


            final_speed = closest_frame[3]
            final_sentence = new_sentence[closest_frame[0]].replace('"', '')

            wav_out = inference.infer_sl_final(wav_seg, final_sentence, sample_rate, 1, spk_emb,final_speed, silence_interval)
            final_audio = inference.infer_english(wav_seg, final_sentence, sample_rate, 1, spk_emb, new_length_scale)
            remaining_length = len(new_vocal_audio) - onset1
            actual_length = min(remaining_length, len(wav_out))
            adjusted_wav_out = wav_out[:actual_length]

            new_vocal_audio[onset1:onset1+actual_length] += adjusted_wav_out
            prev_offset = offset1
            i +=1

    adjusted_rms = np.sqrt(np.mean(new_vocal_audio**2))
    rms_ratio = vocal_rms / adjusted_rms
    new_vocal_audio *= rms_ratio
    new_vocal_audio_resampled = librosa.resample(new_vocal_audio, orig_sr=22050, target_sr=sample_bgm)
    new_vocal_audio_resampled[new_vocal_audio_resampled > 1] = 1.0
    new_vocal_audio_resampled[new_vocal_audio_resampled < -1] = -1.0
    new_audio = new_vocal_audio_resampled+bgm_wav


    sf.write(out_fpath.split('.')[0]+'_synvoc.wav', (new_vocal_audio_resampled*32768).astype('int16'), sample_bgm)
    sf.write(out_fpath.split('.')[0] + '_bgm.wav', (new_audio*32786).astype('int16'),sample_bgm)
    print("finish tts model...")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"TTS Execution time: {execution_time} seconds")


if __name__ == '__main__':
    main()