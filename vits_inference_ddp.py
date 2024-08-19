#inferencei
import os
import sys
import argparse
import torch
import resampy
import librosa
import commons
import utils
import json
import openai
import pdb
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from models_ddp_c import SynthesizerTrn
from text import get_symbols
from text import text_to_sequence, cleaned_text_to_sequence
from utils import load_wav_to_torch
from utils import latest_checkpoint_path
from mel_processing import spectrogram_torch, spec_to_mel_torch

from speaker_encoder import SpeakerEncoder

import soundfile as sf

class Inference:
    def __init__(self, ckpt_path):
        hps = utils.get_hparams_from_file('ckpts/config_ddp.json')
        self.symbols = get_symbols(hps.data.expanded, hps.data.korean)
        self.net_g = SynthesizerTrn(
            len(self.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            n_languages=hps.data.n_languages,
            **hps.model).cuda()
        _ = self.net_g.eval()
        self.se_enc = SpeakerEncoder()
        self.hps = hps
        _ = utils.load_checkpoint(ckpt_path, self.net_g, None)
    
    def get_text(self, text, lidx):
        if lidx == 1:
            text_norm = text_to_sequence(text, ['korean_cleaners'], self.symbols)
        else :
            text_norm = text_to_sequence(text, ['english_cleaners2'], self.symbols)
        hps = self.hps
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_spk_emb(self, waveform):
        waveform = torch.FloatTensor(waveform).type(torch.float32)
        return self.se_enc(waveform)

    def infer_origin(self, waveform, text, sr, lidx=None, origin_frame=None):
        try:
            sequence = self.get_text(text, lidx)
        except Exception as e:
            print(f"An error occurred : {e}")
            pdb.set_trace()
        if sr!=16000:
            waveform = resampy.resample(waveform, sr, 16000)
        duration = origin_frame
        spk_emb = self.get_spk_emb(waveform).detach().cuda()
        with torch.no_grad():
            sequence = sequence.cuda().unsqueeze(0)
            sequence_lengths = torch.LongTensor([sequence.size(1)]).cuda()
            lidx = torch.LongTensor([int(lidx)]).cuda()
            new_length_scale, current_frames = self.net_g.infer_origin(sequence, sequence_lengths, spk=spk_emb, lidx=lidx, duration=duration, noise_scale=.667, noise_scale_w=0.8, length_scale=1.)
        return new_length_scale, current_frames, spk_emb

    def infer_origin_en(self, spk_emb, text, sr, lidx=None, origin_frame=None):
        try:
            sequence = self.get_text(text, lidx)
        except Exception as e:
            print(f"An error occurred : {e}")
            pdb.set_trace()
        duration = origin_frame
        with torch.no_grad():
            sequence = sequence.cuda().unsqueeze(0)
            sequence_lengths = torch.LongTensor([sequence.size(1)]).cuda()
            lidx = torch.LongTensor([int(lidx)]).cuda()
            new_length_scale_en, current_frames_en = self.net_g.infer_origin(sequence, sequence_lengths, spk=spk_emb, lidx=lidx, duration=duration, noise_scale=.667, noise_scale_w=0.8, length_scale=1.)
        return round(new_length_scale_en, 2), current_frames_en

    def infer_t(self, waveform, text, kr_text, sr, lidx=None, new_length_scale=None, origin_frame=None, model=None, spk_emb= None):
        with open('config.json') as f:
            config = json.load(f)
        openai.api_key = config['OPENAI_API_KEY']
        engine = config['OPENAI_ENGINE']
        generated_text = text
        ######원본문장의 sentence transformer
        sentence_embedding_original = model.encode(text)
        print(generated_text)

        try:
            sequence = self.get_text(generated_text, lidx)
        except:
            pdb.set_trace()
        with torch.no_grad():
            sequence = sequence.cuda().unsqueeze(0)
            sequence_lengths = torch.LongTensor([sequence.size(1)]).cuda()
            lidx = torch.LongTensor([int(lidx)]).cuda()
            judge, lengths,w_sum = self.net_g.infer_text(origin_frame, sequence, sequence_lengths, spk=spk_emb, lidx=lidx, noise_scale=.667, noise_scale_w=0.8, length_scale=new_length_scale)
            if judge == False : 
                for i in range(60):
                    if judge :
                        break
                    if lengths :
                        response = openai.Completion.create(
                                        engine=engine,
                                        prompt=f"Shorten the following sentence while maintaining the meaning of the original sentence, but not by more than 10 characters and do not remove conjunctions.:'{generated_text}'",
                                        max_tokens=100,
                                        n=3,
                                        temperature=1,
                                        top_p=1
                                    )
                        sentence_embedding0 = model.encode(response.choices[0].text.strip())
                        sentence_embedding1 = model.encode(response.choices[1].text.strip())
                        sentence_embedding2 = model.encode(response.choices[2].text.strip())
                        similarity0 = cosine_similarity([sentence_embedding_original], [sentence_embedding0])[0][0]
                        similarity1 = cosine_similarity([sentence_embedding_original], [sentence_embedding1])[0][0]
                        similarity2 = cosine_similarity([sentence_embedding_original], [sentence_embedding2])[0][0]
                        sim_max = [similarity0, similarity1, similarity2]
                        a = sim_max.index(max(sim_max))
                        final_similarity = cosine_similarity([sentence_embedding_original],  [model.encode(response.choices[int(a)].text.strip())])[0][0]
                        if final_similarity < 0.75:
                            print('similarity 0.75')
                            continue
                        print('similarity 0.75')
                        generated_text=response.choices[int(a)].text.strip()
                        try:
                            sequence = self.get_text(generated_text, lidx)
                        except:
                            pdb.set_trace()
                        sequence = sequence.cuda().unsqueeze(0)
                        sequence_lengths = torch.LongTensor([sequence.size(1)]).cuda()
                        judge, lengths,w_sum = self.net_g.infer_text(origin_frame, sequence, sequence_lengths, spk=spk_emb, lidx=lidx, noise_scale=.667, noise_scale_w=0.8, length_scale=new_length_scale)
                    else :
                        response = openai.Completion.create(
                                        engine=engine,
                                        prompt=f"Lengthen the following sentence while maintaining the meaning of the original sentence, but not by more than 10 characters and do not remove conjunctions.:'{generated_text}'",
                                        max_tokens=100,
                                        n=3,
                                        temperature=1,
                                        top_p=1
                                    )
                        sentence_embedding0 = model.encode(response.choices[0].text.strip())
                        sentence_embedding1 = model.encode(response.choices[1].text.strip())
                        sentence_embedding2 = model.encode(response.choices[2].text.strip())
                        similarity0 = cosine_similarity([sentence_embedding_original], [sentence_embedding0])[0][0]
                        similarity1 = cosine_similarity([sentence_embedding_original], [sentence_embedding1])[0][0]
                        similarity2 = cosine_similarity([sentence_embedding_original], [sentence_embedding2])[0][0]
                        sim_max = [similarity0, similarity1, similarity2]
                        a = sim_max.index(max(sim_max))
                        final_similarity = cosine_similarity([sentence_embedding_original],  [model.encode(response.choices[int(a)].text.strip())])[0][0]
                        if final_similarity < 0.75:
                            print('similarity 0.75')
                            continue
                        print('similarity 0.75')
                        generated_text=response.choices[int(a)].text.strip()
                        try:
                            sequence = self.get_text(generated_text, lidx)
                        except Exception as E :
                            print(E)
                            pdb.set_trace()
                        sequence = sequence.cuda().unsqueeze(0)
                        sequence_lengths = torch.LongTensor([sequence.size(1)]).cuda()
                        judge, lengths,w_sum = self.net_g.infer_text(origin_frame, sequence, sequence_lengths, spk=spk_emb, lidx=lidx, noise_scale=.667, noise_scale_w=0.8, length_scale=new_length_scale)
                return generated_text, w_sum
            else : 
                return generated_text, w_sum

    def infer_kr(self, waveform, text, sr, lidx=None,new_length_scale=None, spk_emb=None):
        try:
            sequence = self.get_text(text, lidx)
        except Exception as e:
            print(f"An error occurred : {e}")
            pdb.set_trace()
        with torch.no_grad():
            sequence = sequence.cuda().unsqueeze(0)
            sequence_lengths = torch.LongTensor([sequence.size(1)]).cuda()
            lidx = torch.LongTensor([int(lidx)]).cuda()
            kr_list= self.net_g.infer_korean(sequence, sequence_lengths, spk=spk_emb, lidx=lidx, noise_scale=.667, noise_scale_w=0.8, length_scale=new_length_scale)
        return kr_list

    def infer_vowel(self, waveform, text, sr, lidx=None, new_length_scale=None, spk_emb=None, kr_list=None, rep = None, i = None, origin_frame= None):
        sequence = self.get_text(text, lidx)
        with torch.no_grad():
            sequence = sequence.cuda().unsqueeze(0)
            sequence_lengths = torch.LongTensor([sequence.size(1)]).cuda()
            lidx = torch.LongTensor([int(lidx)]).cuda()
            score,path = self.net_g.infer_score(sequence, sequence_lengths, rep = rep, kr_list=kr_list, spk=spk_emb, lidx=lidx, noise_scale=.667, noise_scale_w=0.8, length_scale=new_length_scale, origin_frame=origin_frame)
        return score,path

    def infer_sl_final(self,waveform, text, sr, lidx=None,spk_emb=None, final_speed=None, silence=None):
        sequence = self.get_text(text, lidx)
        with torch.no_grad():
            sequence = sequence.cuda().unsqueeze(0)
            sequence_lengths = torch.LongTensor([sequence.size(1)]).cuda()
            lidx = torch.LongTensor([int(lidx)]).cuda()
            audio = self.net_g.infer_sl(sequence, sequence_lengths, silence=silence, spk=spk_emb, lidx=lidx, noise_scale=.667, noise_scale_w=0.8, length_scale=final_speed)[0][0,0].data.cpu().float().numpy()
        return audio


    def infer(self, waveform, text, sr, lidx=None, spk_emb=None,length_scale = None):
        try:
            sequence = self.get_text(text, lidx)
        except Exception as e:
            print(f"An error occurred : {e}")
        if sr!=16000:
            waveform = resampy.resample(waveform, sr, 16000)
        spk_emb = self.get_spk_emb(waveform).detach().cuda()
        sequence = sequence.cuda().unsqueeze(0)
        sequence_lengths = torch.LongTensor([sequence.size(1)]).cuda()
        lidx = torch.LongTensor([int(lidx)]).cuda()
        audio = self.net_g.infer(sequence, sequence_lengths, spk=spk_emb, lidx=lidx, noise_scale=.667, noise_scale_w=0.8, length_scale=1.3)[0][0,0].data.cpu().float().numpy()
        return audio

    def infer_english(self, waveform, text, sr, lidx=None, spk_emb=None, new_length_scale = None):
        try:
            sequence = self.get_text(text, lidx)
        except Exception as e:
            print(f"An error occurred : {e}")
            import pdb
            pdb.set_trace()
        with torch.no_grad():
            sequence = sequence.cuda().unsqueeze(0)
            sequence_lengths = torch.LongTensor([sequence.size(1)]).cuda()
            lidx = torch.LongTensor([int(lidx)]).cuda()
            audio = self.net_g.infer(sequence, sequence_lengths, spk=spk_emb, lidx=lidx, noise_scale=.667, noise_scale_w=0.8, length_scale=new_length_scale)[0][0,0].data.cpu().float().numpy()
        return audio



if __name__ == '__main__':
    inference = Inference('ckpts/G_ddp_496000.pth')
    waveform, sr = librosa.load('./5258_G2A2E7_HSM_000051.wav', mono=True)
    lid=1
    wav_out = inference.infer(waveform,"여러분, 먼저 입구에 위치한 이곳은 경주 유적지의 전경을 재현한 모형입니다. 이 유적지는 신라 시대에 건설되었으며, 한반도의 중요한 역사적 중심지로서 유명합니다. 이 모형을 통해 당시의 모습을 생생하게 느낄 수 있습니다.", sr,lid)
    sf.write(f'./ct_sample/sample7.wav', wav_out, sr)
    print(wav_out)
