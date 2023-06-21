import numpy as np
import librosa
import pydub
from typing import List, Dict, Union
import random
from pydub import AudioSegment
from pydub import effects
from tensorblur.gaussian import GaussianBlur
# ==================================
def cal_midpoints (dataObj) :

    data = dataObj.get_array_of_samples()    
    data = np.array(data)
    Fs = dataObj.frame_rate

    length     = len(data)
    step       = int(Fs*0.025) # 25ms
    window     = int(Fs*2.000) # 1.0s

    if dataObj.duration_seconds < 100 :
        return []

    start_idx = 0
    stop_idx  = window
    range_    = int((length-window)/step)
    mag = []
    for i in range(range_) :
        if i < 100  or i > (range_ - 10):
             mag.append(0)
        else :
             mag.append(np.sum(abs(data[start_idx:stop_idx])))
        start_idx += step
        stop_idx += step

    min_mag = min(mag[1800:4200])
    max_mag = max(mag[1800:4200])
    th = 0.50*(max_mag - min_mag) + min_mag

    mag_dsc = np.logical_xor(mag[:-1]>th, mag[1:]<th)

    dsc_list = np.argwhere(mag_dsc == False).flatten().tolist()

    last = len(dsc_list)
    tmp = []
    if last%2 == 0 and last <= 10:  # Even number of pairs
        for j in range(int(last/2)):
            if (dsc_list[2*j+1] - dsc_list[2*j]) > 20:
                tmp.append(dsc_list[2*j])
                tmp.append(dsc_list[2*j+1])
    else :
        tmp = dsc_list

    dsc_list = tmp

    last = len(dsc_list)

    midpoints = []
    
    prev_idx = 0
    for j in range(int(last/2)):
        idx = int((dsc_list[2*j] + dsc_list[2*j+1])/2)
        
        diff = step*(idx - prev_idx)/Fs
        if (j == 0) :
            if (diff < 35) or (diff > 105) :
                print ("distance error", j, diff)
                return midpoints
        elif ((diff < 9) or (diff > 16)):
            print ("distance error", j, diff)
            return midpoints
            
        prev_idx = idx
        
    for j in range(int(last/2)):
        midpoints.append(int((dsc_list[2*j] + dsc_list[2*j+1])/2))

    return midpoints


# ==================================
def gen_mel_feature (data, Fs, n_fft, hop_length, win_length, n_mels, log=True) :
    # We should keep power of two for n_fft
    # Fs is 48000, 20 ms is 48*20 = 960
    # Window is 48*50 = 2400
    # fft_step   = 20.0/1000.  # 20ms
    # fft_window = 50.0/1000.  # 50ms
    spectra_abs_min = 0.01 # From Google paper, seems justified

    melspectra = librosa.feature.melspectrogram(y=data, sr=Fs, n_fft=n_fft, hop_length=hop_length, 
                                         win_length=win_length, window='hann', n_mels=n_mels)
    if log :
        mel_log = np.log( np.maximum(spectra_abs_min, melspectra ))
    else :
        mel_log  = melspectra
    mel_log = mel_log.T
    return mel_log


#==========================================
def read_data (dataObj:pydub.AudioSegment, midPoint:List[int], data_list: List,
               duration:int = 10, n_fft:int = 4096, n_mels:int = 96, blur:bool = False, blur_size:int = None) -> List[np.ndarray]:
    data = dataObj.get_array_of_samples() 
    data = np.array(data)
    max_data = 32768 #np.max(np.abs(data))

    Fs      = dataObj.frame_rate
    step    = int(Fs*0.025) # 25ms
    hop_length = step
    win_length = n_fft
    for idx in midPoint:
        for j in range(4) :
            i = random.randrange(0,10)

            start_idx = int(step*idx + Fs/4) - int((duration/2 + 0.034*i)*Fs)
            stop_idx  = start_idx + int(duration*Fs)
            mel_feature = gen_mel_feature (data [start_idx:stop_idx]/max_data, Fs, n_fft, hop_length, win_length, n_mels)
            if (blur):
                blur = GaussianBlur(size=blur_size)
                # mel_feature = np.array(mel_feature)
                mel_feature = np.expand_dims(mel_feature, 2)
                mel_feature = blur.apply(mel_feature)
                mel_feature = np.array(mel_feature) #from eager tensor to np array
                mel_feature = np.squeeze(mel_feature, 2)
            data_list.append(mel_feature)

    return data_list

