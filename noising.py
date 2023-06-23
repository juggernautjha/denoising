def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


import pydub
# import derivative2_train_data  as derivative
import os, sys, glob, random
import numpy as np
import pydub
import typing
from typing import List, Dict, Union, Tuple
import json
if is_interactive():
    from tqdm.notebook import tqdm
else: from tqdm import tqdm
from acoustics import generator


COLORS = ['white', 'blue', 'pink', 'brown', 'violet']

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


def chop_overlay_spit(input_file : str, window_size : int, colors : List[str] = ['white',  'blue', 'pink', 'brown', 'violet']) -> Tuple:
    '''
    Takes in an input file, splits it into multiple (possibly overlapping windows) of size window_size, and then overlays some noise (randomly picked from the available noise types)
    '''
    clear = []
    noisy = []
    orig_sound = pydub.AudioSegment.from_file(input_file, format="flac")
    orig_sound = orig_sound.get_array_of_samples()
    diff = window_size - len(orig_sound)%window_size 
    orig_sound = list(orig_sound) + [0]*diff
    orig_sound = np.array(orig_sound)
    # orig_sound = orig_sound.concat([0]*diff)
    orig_chunks = np.split(orig_sound, len(orig_sound)//window_size)    
    for i in tqdm(orig_chunks):
        clear.append(i)
        color = random.randint(0, len(colors)-1)
        noise = generator.noise(window_size,color=colors[color])
        noisy.append(i+noise)
    # print(len(clear), len(noisy))
    return clear, noisy







if __name__ == '__main__':
    INPUT_DIR = 'samples'
    files = glob.glob(f'{INPUT_DIR}/*.flac')
    for i in files: chop_overlay_spit(i, 49000)