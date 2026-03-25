
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

import numpy as np

def extract_landmarks(frame, num_landmarks=68):
    # Placeholder for facial landmark extraction
    return np.zeros((num_landmarks, 2))

def normalize_landmarks(landmarks, frame_shape):
    h, w = frame_shape[:2]
    normalized = landmarks.copy().astype(float)
    normalized[:, 0] /= w
    normalized[:, 1] /= h
    return normalized

import numpy as np

def align_modalities(audio_features, video_features, audio_fps=16000, video_fps=25):
    ratio = audio_fps / video_fps
    aligned = []
    for i in range(len(video_features)):
        audio_idx = int(i * ratio)
        if audio_idx < len(audio_features):
            aligned.append((audio_features[audio_idx], video_features[i]))
    return aligned

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

import numpy as np

def extract_landmarks(frame, num_landmarks=68):
    # Placeholder for facial landmark extraction
    return np.zeros((num_landmarks, 2))

def normalize_landmarks(landmarks, frame_shape):
    h, w = frame_shape[:2]
    normalized = landmarks.copy().astype(float)
    normalized[:, 0] /= w
    normalized[:, 1] /= h
    return normalized

import numpy as np

def align_modalities(audio_features, video_features, audio_fps=16000, video_fps=25):
    ratio = audio_fps / video_fps
    aligned = []
    for i in range(len(video_features)):
        audio_idx = int(i * ratio)
        if audio_idx < len(audio_features):
            aligned.append((audio_features[audio_idx], video_features[i]))
    return aligned

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

import numpy as np

def extract_landmarks(frame, num_landmarks=68):
    # Placeholder for facial landmark extraction
    return np.zeros((num_landmarks, 2))

def normalize_landmarks(landmarks, frame_shape):
    h, w = frame_shape[:2]
    normalized = landmarks.copy().astype(float)
    normalized[:, 0] /= w
    normalized[:, 1] /= h
    return normalized

import numpy as np

def align_modalities(audio_features, video_features, audio_fps=16000, video_fps=25):
    ratio = audio_fps / video_fps
    aligned = []
    for i in range(len(video_features)):
        audio_idx = int(i * ratio)
        if audio_idx < len(audio_features):
            aligned.append((audio_features[audio_idx], video_features[i]))
    return aligned

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

import numpy as np

def extract_landmarks(frame, num_landmarks=68):
    # Placeholder for facial landmark extraction
    return np.zeros((num_landmarks, 2))

def normalize_landmarks(landmarks, frame_shape):
    h, w = frame_shape[:2]
    normalized = landmarks.copy().astype(float)
    normalized[:, 0] /= w
    normalized[:, 1] /= h
    return normalized

import numpy as np

def align_modalities(audio_features, video_features, audio_fps=16000, video_fps=25):
    ratio = audio_fps / video_fps
    aligned = []
    for i in range(len(video_features)):
        audio_idx = int(i * ratio)
        if audio_idx < len(audio_features):
            aligned.append((audio_features[audio_idx], video_features[i]))
    return aligned

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

import numpy as np

def extract_landmarks(frame, num_landmarks=68):
    # Placeholder for facial landmark extraction
    return np.zeros((num_landmarks, 2))

def normalize_landmarks(landmarks, frame_shape):
    h, w = frame_shape[:2]
    normalized = landmarks.copy().astype(float)
    normalized[:, 0] /= w
    normalized[:, 1] /= h
    return normalized
