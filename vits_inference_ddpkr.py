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
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from models_ddp_ckr import SynthesizerTrn
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
            text_norm = text_to_sequence(text, ['korean_cleaners2'], self.symbols)
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

    def infer_t(self, waveform, text, sr, lidx=None, new_length_scale=None, origin_frame=None, model=None, spk_emb= None, en_original=None):
        with open('config.json') as f:
            config = json.load(f)
        openai.api_key = config['OPENAI_API_KEY']
        engine = config['OPENAI_ENGINE']
        generated_text = text
        sentence_embedding_original = model.encode(en_original)
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
                                        prompt=f"원문의 의미를 유지하면서 다음 문장을 단축하되, 10자 이내로 줄이지 말고 접속사를 제거하지 않는다.:'{generated_text}'",
                                        max_tokens=200,
                                        n=3,
                                        temperature=0.8,
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
                                        prompt=f"원래 문장의 의미를 유지하면서 다음 문장을 길게 하되 10자 이내로 하고 접속사를 제거하지 않습니다.:'{generated_text}'",
                                        max_tokens=200,
                                        n=3,
                                        temperature=0.8,
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

    def infer_en(self, waveform, text, sr, lidx=None,new_length_scale=None, spk_emb=None):
        try:
            sequence = self.get_text(text, lidx)
        except Exception as e:
            print(f"An error occurred : {e}")
            pdb.set_trace()
        with torch.no_grad():
            sequence = sequence.cuda().unsqueeze(0)
            sequence_lengths = torch.LongTensor([sequence.size(1)]).cuda()
            lidx = torch.LongTensor([int(lidx)]).cuda()
            en_list= self.net_g.infer_ensh(sequence, sequence_lengths, spk=spk_emb, lidx=lidx, noise_scale=.667, noise_scale_w=0.8, length_scale=new_length_scale)
        return en_list

    def infer_vowel(self, waveform, text, sr, lidx=None, new_length_scale=None, spk_emb=None, en_list=None, rep = None, i = None, origin_frame= None):
        sequence = self.get_text(text, lidx)
        with torch.no_grad():
            sequence = sequence.cuda().unsqueeze(0)
            sequence_lengths = torch.LongTensor([sequence.size(1)]).cuda()
            lidx = torch.LongTensor([int(lidx)]).cuda()
            score,path = self.net_g.infer_score(sequence, sequence_lengths, rep = rep, kr_list=en_list, spk=spk_emb, lidx=lidx, noise_scale=.667, noise_scale_w=0.8, length_scale=new_length_scale, origin_frame=origin_frame)
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
            import pdb
            pdb.set_trace()
        sequence = sequence.cuda().unsqueeze(0)
        sequence_lengths = torch.LongTensor([sequence.size(1)]).cuda()
        lidx = torch.LongTensor([int(lidx)]).cuda()
        audio = self.net_g.infer(sequence, sequence_lengths, spk=spk_emb, lidx=lidx, noise_scale=.667, noise_scale_w=0.8, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
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
    model = SentenceTransformer('all-MiniLM-L6-v2')
    waveform, sr = librosa.load('aunion_sample/0307/7549_G1A2E7_LHJ_000677.wav', mono=True)
    lid=0
    generated_text,b = inference.infer_sl_final(waveform,"The U.S. Centers for Disease Control and Prevention also explains that e-cigarettes are not effective in quitting smoking.", sr,lid,spk_emb=None, final_speed=1.19, silence=None)
    i=0
    sf.write(f'aunion_sample/0402/english_test_lengthswhy_____{i}.wav', generated_text, sr)
    print(generated_text)
    print(b)