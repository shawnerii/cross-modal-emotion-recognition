import gradio as gr
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# === Load trained models ===
cnn_model = tf.keras.models.load_model("cnn_model.h5")
lstm_model = tf.keras.models.load_model("lstm_model.h5")
clstm_model = tf.keras.models.load_model("clstm_model.h5")

# === Label encoder ===
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
encoder = LabelEncoder()
encoder.fit(emotion_labels)

# === Audio preprocessing ===
def extract_mfcc_from_audio(file_path, sample_rate=22050, n_mfcc=40, max_len=174):
    signal, sr = librosa.load(file_path, sr=sample_rate, duration=3.0)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

# === Prediction handler ===
def predict_emotion(audio):
    mfcc = extract_mfcc_from_audio(audio)
    
    # CNN expects shape (40, 174, 1)
    x_cnn = mfcc[np.newaxis, ..., np.newaxis]
    pred_cnn = cnn_model.predict(x_cnn)[0]

    # LSTM/CLSTM expects shape (174, 40)
    x_seq = np.transpose(mfcc)[np.newaxis, ...]
    pred_lstm = lstm_model.predict(x_seq)[0]
    pred_clstm = clstm_model.predict(x_seq)[0]

    idx_cnn = np.argmax(pred_cnn)
    idx_lstm = np.argmax(pred_lstm)
    idx_clstm = np.argmax(pred_clstm)

    return (
    f"{encoder.inverse_transform([idx_cnn])[0]} ({pred_cnn[idx_cnn]*100:.2f}%)",
    f"{encoder.inverse_transform([idx_lstm])[0]} ({pred_lstm[idx_lstm]*100:.2f}%)",
    f"{encoder.inverse_transform([idx_clstm])[0]} ({pred_clstm[idx_clstm]*100:.2f}%)"
    )

# === Gradio UI ===
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(type="filepath", label="Upload or record audio"),
    outputs=[
        gr.Text(label="CNN Prediction"),
        gr.Text(label="LSTM Prediction"),
        gr.Text(label="CLSTM Prediction")
    ],
    title="Speech Emotion Recognition",
    description="Upload a WAV file (3 sec) to classify emotion using CNN, LSTM, and CLSTM models"
)

if __name__ == "__main__":
    interface.launch()