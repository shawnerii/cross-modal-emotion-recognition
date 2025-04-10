
import numpy as np

def normalize_audio(signal, target_rms=0.1):
    rms = np.sqrt(np.mean(signal**2))
    if rms == 0:
        return signal
    return signal * (target_rms / rms)

def frame_signal(signal, frame_len=512, hop=256):
    frames = []
    for i in range(0, len(signal) - frame_len, hop):
        frames.append(signal[i:i+frame_len])
    return np.array(frames)
