
from pathlib import Path
import numpy as np
from wrapper import PengiWrapper as Pengi
from msclap import CLAP
import librosa
from transformers import AutoProcessor, Wav2Vec2Model
import torch
from transformers import AutoFeatureExtractor, WavLMForXVector
from tqdm import tqdm 

def get_pengi_embed(folder):
    pengi = Pengi(config="base_no_text_enc")
    files = Path(folder).glob("**/*.WAV")
    
    for file in tqdm(files, total=len(files)):
        
        _, audio_embeddings = pengi.get_audio_embeddings(audio_paths=file)
        
        np.save(f"{file.replace(".WAV", ".npy")}", audio_embeddings)
        
def get_clap_embed(folder):
    clap_model = CLAP(version = '2023', use_cuda=False)
    files = Path(folder).glob("**/*.WAV")
    
    for file in tqdm(files, total=len(files)):
        
        audio_embeddings = clap_model.get_audio_embeddings(file)
        
        np.save(f"{file.replace(".WAV", ".npy")}", audio_embeddings)
    
def get_wavllm_embed(folder):
    files = Path(folder).glob("**/*.WAV")
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")

    for file in tqdm(files, total=len(files)):
        wav, sr = librosa.load(file, sr=16_000)
        inputs = feature_extractor(wav, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            audio_embeddings = model(**inputs).embeddings

        audio_embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

                
        
        np.save(f"{file.replace(".WAV", ".npy")}", audio_embeddings)
        


    
def get_wav2vec_embed(folder):
    files = Path(folder).glob("**/*.WAV")
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    
    for file in tqdm(files, total=len(files)):
        wav, sr = librosa.load(file, sr=16_000)
        inputs = processor(wav, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            audio_embeddings = model(**inputs)

        audio_embeddings = audio_embeddings.last_hidden_state
        
        np.save(f"{file.replace(".WAV", ".npy")}", audio_embeddings)



