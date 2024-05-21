
from tqdm import tqdm
import os
import glob
import numpy as np
from pathlib import Path
from elevenlabs import clone, generate, play, save
import soundfile as sf
import time

from elevenlabs import set_api_key

