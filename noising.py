import pydub
# import derivative2_train_data  as derivative
import os, sys, glob, random
import numpy as np
import pydub
import typing
from typing import List, Dict, Union, Tuple
import json
from tqdm.notebook import tqdm
from acoustics import generator


COLORS = ['white', 'blue', 'pink', 'brown', ]

def overlay_noise(input_file : str, quiet_by_dB : int, color : str = 'white', pad_to :int = 140) -> Tuple:
    orig_sound = pydub.AudioSegment.from_file(input_file, format="flac")
    if len(orig_sound) < pad_to*1000:
            silence = pydub.AudioSegment.silent(pad_to*1000 - len(orig_sound))
            orig_sound = orig_sound + silence
    elif len(orig_sound) > pad_to*1000:
        orig_sound = orig_sound[pad_to*1000]
    orig = orig_sound.get_array_of_samples()
    num = len(orig)
    frame_rate = orig_sound.frame_rate
    orig_sound = pydub.effects.normalize(orig_sound)
    audio_len  = len(orig_sound)  
    noise = generator.noise(num,color=color)
    noise = pydub.AudioSegment(noise.tobytes(), frame_rate = frame_rate, channels = 1, sample_width = 2)
    noise = noise - quiet_by_dB
    new_sound = orig_sound.overlay(noise)
    new_sound = new_sound.set_sample_width(2)
    orig_sound = orig_sound.normalize()
    new_sound = new_sound.normalize()
    new = new_sound.get_array_of_samples()
    min_len = min(len(orig), len(new))
    return new[:min_len], orig[:min_len] 

def save_overlaid_noise(input_file : str, quiet_by_dB : int, output_dir : str, color : str = 'white', pad_to : int = 140) -> str:
    '''
    Utility function.
    Overlay sound samples with white noise for autoencoder training. Returns the path of the saved file.
    '''
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    prefix_output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0])
    orig_sound = pydub.AudioSegment.from_file(input_file, format="flac")
    if len(orig_sound) < pad_to*1000:
            silence = pydub.AudioSegment.silent(pad_to*1000 - len(orig_sound))
            orig_sound = orig_sound + silence
    elif len(orig_sound) > pad_to*1000:
        orig_sound = orig_sound[pad_to*1000]
    orig = orig_sound.get_array_of_samples()
    num = len(orig)
    frame_rate = orig_sound.frame_rate
    orig_sound = pydub.effects.normalize(orig_sound)
    noise = generator.noise(num,color=color)
    noise = pydub.AudioSegment(noise.tobytes(), frame_rate = frame_rate, channels = 1, sample_width = 2)
    noise = noise - quiet_by_dB
    new_sound = orig_sound.overlay(noise)
    new_sound = new_sound.set_sample_width(2)
    new_sound_file = prefix_output_file + f"_{color}{random.randint(1,75)}_" + ".flac"
    new_sound.export(new_sound_file, format='flac')
    return new_sound_file   


def save_overlaid_dataset(input_files : List[str], num_samples : int, quiet_by_dB : int, outfile : str, output_dir : str, pad_to : int = 140) -> None:
    dataset = {
        i : [] for i in input_files
    }
    for path in tqdm(input_files):
        colors = random.choices(COLORS, k = num_samples)
        for color in tqdm(colors):
            dataset[path].append(save_overlaid_noise(path, quiet_by_dB, output_dir, color, pad_to))
    f = open(outfile, 'w')
    json.dump(dataset, f)
    return

if __name__ == '__main__':
    INPUT_DIR = 'samples'
    files = glob.glob(f'{INPUT_DIR}/*.flac')
    save_overlaid_dataset(files, 3, 15, 'dataset.json', 'overlaid')