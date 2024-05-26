
from tqdm import tqdm
import os
import glob
import numpy as np
from pathlib import Path
from elevenlabs import clone, generate, play, save
import soundfile as sf
import time
from pathlib import Path 

from elevenlabs import set_api_key

def synth_elabs(folder):
    API_KEY = "Your_key"
    set_api_key(API_KEY)
    

    spkrs = Path(folder).glob("**/")

    counter =  0
    for spkr in tqdm(spkrs, total= len(spkrs)):

        files = glob.glob(f'{spkr}/*.WAV')

        # Create a synthetic voice

        voice = clone(
            name= spkr,
            files=files,)

        # Generate utterance for the synthetic voice

        for file in files:


            with open(file.replace(".WAV", ".TXT"), 'r') as f:
                trans =[f.read().strip().split(" ", 2)[-1]]


            for t in trans:


                text = t
                wav_file = file.rsplit('/', 1)[-1].replace('.WAV', '')


                out_path = file.replace('.WAV', '_elabs.WAV')


                audio = generate(text=text, voice=voice)
                save(audio, out_path)

        counter += 1
        # to avoide too many requests error
        if counter> 60:
            time.sleep(80)
            counter =  0

  
    
def synth_mqtts(folder):
    return None
    
    
def synth_yourtts(folder):
    return None

def synth_xtts(folder):
    return None
    
    