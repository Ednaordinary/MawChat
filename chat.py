import time
import torch
import gc
from transformers import AutoModelForCausalLM, TextIteratorStreamer, AutoTokenizer
import whisper
import speech_recognition as sr
import threading
import random
import numpy as np
import nltk
from queue import Queue
from munch import Munch
from torch import nn
import torch.nn.functional as F
import torchaudio
import yaml
from models import *
from utils import *
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from nltk.tokenize import word_tokenize
import phonemizer
import sounddevice as sd
import datetime

# model_mode = "medium" # important! Medium is faster but may have worse transcriptions
model_mode = "large" # important! Large has a lot more latency but better transcriptions

past_chat = []
current_chat = None
current_response = None
model_loaded = 0 # 0 is unloaded, 1 is a request to load, 2 is loaded, 3 is a request to unload, 4 is to move to cpu, 5 is to move to gpu
audio_text_list = []
audio_play_list = []

def text_streamer(streamer):
    """
    For streaming text while model is generating.
    """
    global audio_text_list
    sentence_cache = ""
    for word in streamer:
        print(word, flush=True, end='')
        if "." in word:
            sentence_cache = sentence_cache + word.split(".")[0]
            audio_text_list.append(sentence_cache)
            sentence_cache = word.split(".")[1]
        elif "!" in word:
            sentence_cache = sentence_cache + word.split("!")[0]
            audio_text_list.append(sentence_cache)
            sentence_cache = word.split("!")[1]
        elif "?" in word:
            sentence_cache = sentence_cache + word.split("?")[0]
            audio_text_list.append(sentence_cache)
            sentence_cache = word.split("?")[1]
        else:
            sentence_cache = sentence_cache + word
    print("\n")
    if sentence_cache != "":
        audio_text_list.append(sentence_cache)

def model_runner():
    """
    This will be a thread for running the llama3 model. Should dynamically offload if we need vram
    """
    global past_chat
    global current_chat
    global current_response
    global model_loaded
    model = None
    tokenizer = None
    while True:
        time.sleep(0.01)
        if model_loaded == 1:
            model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3-8B-Instruct",
                #device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            )
            if model_mode == "large":
                model.to('cpu')
            tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Meta-Llama-3-8B-Instruct",
                    truncation_side="left",
            )
            model_loaded = 2
        if model_loaded == 3:
            model, tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
        #if model_loaded == 4:
        #    model.to('cpu')
        #    model_loaded = 2
        #if model_loaded == 5:
        #    model.to('cuda')
        #    model_loaded = 2
        if current_chat:
            if model_loaded != 2:
                model_loaded = 1
            else:
                if model.device.type == 'cpu':
                    model.to('cuda')
                sys_prompt= [ {"role": "system", "content": "You are Maw, an intelligence model that answers questions to the best of my knowledge. You may also be referred to as Mode Assistance. You were developed by Mode LLC, a company founded by Edna. Respond briefly, as your words are spoken out loud."}]
                this_chat = {"role": "user", "content": str(current_chat)}
                past_chat.append(this_chat)
                init_prompt = tokenizer.apply_chat_template(conversation=sys_prompt, tokenize=True, return_tensors='pt', add_generation_prompt=False)
                past_chat.append({"role": "user", "content": str(current_chat)})
                input_ids = tokenizer.apply_chat_template(conversation=past_chat, tokenize=True, return_tensors='pt', add_generation_prompt=True, max_length=7200 - init_prompt.size()[1], truncation=True)
                input_ids = torch.cat((init_prompt, input_ids), 1).to("cuda")
                output_streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                stream_thread = threading.Thread(target=text_streamer, args=[output_streamer])
                model_kwargs = dict(input_ids=input_ids, max_new_tokens=768, use_cache=True,  do_sample=False, pad_token_id=tokenizer.eos_token_id) #, max_matching_ngram_size=2, prompt_lookup_num_tokens=15) #, temperature=0.6, top_p=0.9)
                #TODO: Add prompt_lookup_num_tokens once the eot_id pr is merged
                stop_token = tokenizer.encode("<|eot_id|>")
                stream_thread.start()
                try:
                    current_response = model.generate(**model_kwargs, streamer=output_streamer, eos_token_id=stop_token)
                except:
                    pass
                threading.Thread(target=unload_llama, args=[model]).start()

def unload_llama(model):
    #this blocks StyleTTS2 from running if we are transferring models so we need to wait for it
    global current_chat
    gc.collect()
    torch.cuda.empty_cache()
    if model_mode == "large":
        while audio_text_list != []:
            time.sleep(0.01)
        model.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
    current_chat = None

def speaker_runner():
    """
    For running StyleTTS2. Basically all of this is borrowed from the LJSpeech notebook.
    """
    device = "cuda"
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(0)
    np.random.seed(0)
    nltk.download('punkt')
    text_cleaner = TextCleaner()
    to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    mean, std = -4, 4
    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask
    def preprocess(wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor

    def compute_style(ref_dicts):
        reference_embeddings = {}
        for key, path in ref_dicts.items():
            wave, sr = librosa.load(path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                audio = librosa.resample(audio, sr, 24000)
            mel_tensor = preprocess(audio).to(device)

            with torch.no_grad():
                ref = model.style_encoder(mel_tensor.unsqueeze(1))
            reference_embeddings[key] = (ref.squeeze(1), audio)

        return reference_embeddings
    
    global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')
    config = yaml.safe_load(open("Models/LJSpeech/config.yml"))
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)
    model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]
    params_whole = torch.load("Models/LJSpeech/epoch_2nd_00100.pth", map_location='cpu')
    params = params_whole['net']
    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                # load params
                model[key].load_state_dict(new_state_dict, strict=False)
    #             except:
    #                 _load(params[key], model[key])
    _ = [model[key].eval() for key in model]
    
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )
    
    def inference(text, noise, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        text = text.replace('"', '')
        ps = global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        tokens = text_cleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)

            t_en = model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = sampler(noise,
                  embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
                  embedding_scale=embedding_scale).squeeze(0)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = model.predictor.lstm(d)
            duration = model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_dur[-1] += 5

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
            out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)),
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()
    
    def LFinference(text, s_prev, noise, alpha=0.7, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        text = text.replace('"', '')
        ps = global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        tokens = text_cleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

        with torch.no_grad():
          input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
          text_mask = length_to_mask(input_lengths).to(tokens.device)

          t_en = model.text_encoder(tokens, input_lengths, text_mask)
          bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
          d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

          s_pred = sampler(noise,
                embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
                embedding_scale=embedding_scale).squeeze(0)

          if s_prev is not None:
              # convex combination of previous and current style
              s_pred = alpha * s_prev + (1 - alpha) * s_pred

          s = s_pred[:, 128:]
          ref = s_pred[:, :128]

          d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

          x, _ = model.predictor.lstm(d)
          duration = model.predictor.duration_proj(x)
          duration = torch.sigmoid(duration).sum(axis=-1)
          pred_dur = torch.round(duration.squeeze()).clamp(min=1)

          pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
          c_frame = 0
          for i in range(pred_aln_trg.size(0)):
              pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
              c_frame += int(pred_dur[i].data)

          # encode prosody
          en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
          F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
          out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)),
                                  F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy(), s_pred
    global audio_text_list
    global model_loaded
    model_loaded = 1 # Don't load llama until after StyleTTS2, otherwise we run into errors.
    while True:
        while audio_text_list == []:
            time.sleep(0.01)
        if audio_text_list[0] != "":
            try:
                start = time.time()
                noise = torch.randn(1,1,256).to(device)
                wav = inference(audio_text_list[0], noise, diffusion_steps=7, embedding_scale=1)
                #print("(StyleTTS2) Real time factor:", round((len(wav) / 24000) / (time.time() - start), 2))
                audio_play_list.append(wav)
            except:
                pass
        audio_text_list.pop(0)

def text_player():
    """
    This ensures all wavs are generated immediately so we do not block at all. Blocking is bad. I don't like blocking.
    """
    global audio_play_list
    while True:
        while audio_play_list == []:
            time.sleep(0.01)
        sd.play(audio_play_list[0], samplerate=24000)
        sd.wait()
        audio_play_list.pop(0)

def text_input():
    global current_chat
    while True:
        current_chat = input("User>")
        while current_chat:
            time.sleep(0.01)

def load_to(model, device):
    model.to(device)
    gc.collect()
    torch.cuda.empty_cache()

def audio_input():
    global current_chat
    global model_loaded
    pt = None
    dq = Queue()
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 1000
    recognizer.dynamic_energy_threshold = False
    mic_name = "default"
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        if mic_name in name:
            source = sr.Microphone(sample_rate=16000, device_index=index)
    if not source:
        print("Failed to find a mic!")
    else:
        with torch.no_grad():
            while model_loaded == 0:
                time.sleep(0.01)
            if model_mode == "large":
                model = whisper.load_model("large", device='cuda')
            elif model_mode == "medium":
                model = whisper.load_model("medium.en", device='cuda')
            print("Model loaded! Start speaking now")
            #model.to('cpu')
            rt = 1.0
            pto = 1.0
            transcript = ['']
            with source:
                recognizer.adjust_for_ambient_noise(source)
            def record_callback(_, audio:sr.AudioData) -> None:
                data = audio.get_raw_data()
                dq.put(data)
            recognizer.listen_in_background(source, record_callback, phrase_time_limit=rt)
            while True:
                now = datetime.datetime.now(datetime.UTC)
                if not dq.empty():
                    phrase_complete = False
                    if pt and now - pt > datetime.timedelta(seconds=pto):
                        phrase_complete = True
                    pt = now
                    audio_data = b''.join(dq.queue)
                    dq.queue.clear()
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    #model_loaded = 4
                    result = model.transcribe(audio_np, fp16=True)
                    #model_loaded = 5
                    text = result['text'].strip()
                    if phrase_complete:
                        transcript.append(text)
                    else:
                        transcript[-1] = text
                    print(text, flush=True, end='')
                    if "." in transcript[-1] or "?" in transcript[-1] or "!" in transcript[-1]:
                        print("\n")
                        if model_mode == "large":
                            threading.Thread(target=load_to, args=[model, 'cpu']).start()
                        current_chat = ''.join(transcript)
                        transcript = ['']
                        while current_chat != None:
                            time.sleep(0.01)
                        if model_mode == "large":
                            model.to('cuda')
                else:
                    time.sleep(0.01)

llama_thread = threading.Thread(target=model_runner)
#text_thread = threading.Thread(target=text_input)
text_player_thread = threading.Thread(target=text_player)
speaker_thread = threading.Thread(target=speaker_runner)
text_player_thread.start()
speaker_thread.start()
llama_thread.start()
#text_thread.start()
#text_thread.join()
audio_input_thread = threading.Thread(target=audio_input)
audio_input_thread.start()
audio_input_thread.join()
