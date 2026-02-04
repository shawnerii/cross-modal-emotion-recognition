
import numpy as np

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class EmotionClassifier:
    def __init__(self, n_classes=7):
        self.n_classes = n_classes
        self.weights = None

    def predict_proba(self, features):
        # Softmax placeholder
        scores = np.random.randn(self.n_classes)
        exp_scores = np.exp(scores - scores.max())
        return exp_scores / exp_scores.sum()

    def predict(self, features):
        proba = self.predict_proba(features)
        return EMOTIONS[np.argmax(proba)]

import numpy as np

def attention_weights(query, keys):
    scores = np.dot(keys, query)
    scores = scores - scores.max()
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum()

def context_vector(weights, values):
    return np.dot(weights, values)

import numpy as np

def late_fusion(audio_probs, video_probs, audio_weight=0.4, video_weight=0.6):
    return audio_weight * audio_probs + video_weight * video_probs

def early_fusion(audio_features, video_features):
    return np.concatenate([audio_features.flatten(), video_features.flatten()])

import numpy as np

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class EmotionClassifier:
    def __init__(self, n_classes=7):
        self.n_classes = n_classes
        self.weights = None

    def predict_proba(self, features):
        # Softmax placeholder
        scores = np.random.randn(self.n_classes)
        exp_scores = np.exp(scores - scores.max())
        return exp_scores / exp_scores.sum()

    def predict(self, features):
        proba = self.predict_proba(features)
        return EMOTIONS[np.argmax(proba)]

import numpy as np

def attention_weights(query, keys):
    scores = np.dot(keys, query)
    scores = scores - scores.max()
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum()

def context_vector(weights, values):
    return np.dot(weights, values)

import numpy as np

def late_fusion(audio_probs, video_probs, audio_weight=0.4, video_weight=0.6):
    return audio_weight * audio_probs + video_weight * video_probs

def early_fusion(audio_features, video_features):
    return np.concatenate([audio_features.flatten(), video_features.flatten()])

import numpy as np

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class EmotionClassifier:
    def __init__(self, n_classes=7):
        self.n_classes = n_classes
        self.weights = None

    def predict_proba(self, features):
        # Softmax placeholder
        scores = np.random.randn(self.n_classes)
        exp_scores = np.exp(scores - scores.max())
        return exp_scores / exp_scores.sum()

    def predict(self, features):
        proba = self.predict_proba(features)
        return EMOTIONS[np.argmax(proba)]

import numpy as np

def attention_weights(query, keys):
    scores = np.dot(keys, query)
    scores = scores - scores.max()
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum()

def context_vector(weights, values):
    return np.dot(weights, values)

import numpy as np

def late_fusion(audio_probs, video_probs, audio_weight=0.4, video_weight=0.6):
    return audio_weight * audio_probs + video_weight * video_probs

def early_fusion(audio_features, video_features):
    return np.concatenate([audio_features.flatten(), video_features.flatten()])

import numpy as np

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class EmotionClassifier:
    def __init__(self, n_classes=7):
        self.n_classes = n_classes
        self.weights = None

    def predict_proba(self, features):
        # Softmax placeholder
        scores = np.random.randn(self.n_classes)
        exp_scores = np.exp(scores - scores.max())
        return exp_scores / exp_scores.sum()

    def predict(self, features):
        proba = self.predict_proba(features)
        return EMOTIONS[np.argmax(proba)]

import numpy as np

def attention_weights(query, keys):
    scores = np.dot(keys, query)
    scores = scores - scores.max()
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum()

def context_vector(weights, values):
    return np.dot(weights, values)
