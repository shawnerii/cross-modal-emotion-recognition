
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
