#!/usr/bin/env python
# coding: utf-8

# In[26]:


## PHASE 1: Dataset Preparation (RAVDESS) ##
# Extract Labels and Save Metadata

import os
import pandas as pd

# Define path to RAVDESS folder
ravdess_path = 'data'

# Emotion code mapping
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Create list to hold metadata
metadata = []

# Traverse RAVDESS folder
for root, _, files in os.walk(ravdess_path):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split('-')
            emotion_code = parts[2]
            actor_id = parts[-1].split('.')[0]  # e.g., 12 from "03-01-05-01-02-01-12.wav"
            emotion = emotion_map.get(emotion_code)
            file_path = os.path.join(root, file)
            metadata.append([file_path, emotion, actor_id])

# Create DataFrame
df = pd.DataFrame(metadata, columns=['path', 'emotion', 'actor'])

# Save metadata to CSV
df.to_csv('ravdess_metadata.csv', index=False)

# Show sample
df.head()


# In[28]:


##  PHASE 2: Audio Feature Extraction (for CNN, LSTM, CLSTM) ##
# Feature Extraction (MFCC only – robust and sufficient for DL)

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Load metadata
df = pd.read_csv('ravdess_metadata.csv')

# Audio processing parameters
SAMPLE_RATE = 22050
DURATION = 3  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MFCC = 40

# Function to extract padded MFCCs
def extract_mfcc(file_path, max_len=174):
    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)
        # Padding/truncating to fixed length
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Lists to hold processed features and labels
X = []
y = []

# Extract features
print("Extracting MFCC features...")
for i in tqdm(range(len(df))):
    file_path = df.iloc[i]['path']
    emotion = df.iloc[i]['emotion']
    mfcc = extract_mfcc(file_path)
    if mfcc is not None:
        X.append(mfcc)
        y.append(emotion)

# Convert to arrays
X = np.array(X)
y = np.array(y)

# Save for future use
np.save('X_mfcc.npy', X)
np.save('y_emotion.npy', y)

# Show shape
print("X shape:", X.shape)  # (num_samples, 40, 174)
print("y shape:", y.shape)


# In[30]:


## PHASE 3: Model Building ##
# 3.1: Label Encoding & Dataset Splitting

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load saved features and labels
X = np.load('X_mfcc.npy')
y = np.load('y_emotion.npy')

# Encode emotion labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Split: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y_onehot, test_size=0.30, stratify=y_onehot, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# Shapes
print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


# In[40]:


# 3.2: CNN Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, BatchNormalization

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Reshape for CNN (add channel dim)
X_train_cnn = X_train[..., np.newaxis]
X_val_cnn = X_val[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

cnn_model = build_cnn_model(X_train_cnn.shape[1:], num_classes=y_train.shape[1])
cnn_model.summary()


# In[42]:


# 3.3: LSTM Model

from tensorflow.keras.layers import LSTM

def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=False, input_shape=input_shape),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

# No reshape needed: shape is (samples, time, features)
X_train_lstm = np.transpose(X_train, (0, 2, 1))  # (samples, time_steps, features)
X_val_lstm = np.transpose(X_val, (0, 2, 1))
X_test_lstm = np.transpose(X_test, (0, 2, 1))

lstm_model = build_lstm_model(X_train_lstm.shape[1:], num_classes=y_train.shape[1])
lstm_model.summary()


# In[44]:


# 3.4: CLSTM (CNN + LSTM)

from tensorflow.keras.layers import Conv1D, MaxPooling1D

def build_clstm_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(128, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        LSTM(128),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

X_train_clstm = np.transpose(X_train, (0, 2, 1))  # time_steps x features
X_val_clstm = np.transpose(X_val, (0, 2, 1))
X_test_clstm = np.transpose(X_test, (0, 2, 1))

clstm_model = build_clstm_model(X_train_clstm.shape[1:], num_classes=y_train.shape[1])
clstm_model.summary()


# In[46]:


## PHASE 4: Model Training ##
# 4.1. Common Setup

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Callback Setup
def get_callbacks(model_name):
    return [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        ModelCheckpoint(f"{model_name}.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
    ]


# In[48]:


# 4.2. Training Function

def train_model(model, X_train, y_train, X_val, y_val, model_name, epochs=100, batch_size=32):
    callbacks = get_callbacks(model_name)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


# In[50]:


# 4.3. Train All Three Models


# In[52]:


cnn_history = train_model(cnn_model, X_train_cnn, y_train, X_val_cnn, y_val, "cnn_model")


# In[54]:


lstm_history = train_model(lstm_model, X_train_lstm, y_train, X_val_lstm, y_val, "lstm_model")


# In[56]:


clstm_history = train_model(clstm_model, X_train_clstm, y_train, X_val_clstm, y_val, "clstm_model")


# In[58]:


# 4.4. Visualize Training Curves (Loss & Accuracy)

import matplotlib.pyplot as plt

def plot_history(history, model_name):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


# In[60]:


plot_history(cnn_history, "CNN")
plot_history(lstm_history, "LSTM")
plot_history(clstm_history, "CLSTM")


# In[62]:


## PHASE 5: Model Evaluation ##
# 5.1: Setup Evaluation Imports

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[64]:


# 5.2: Decode Predictions to Labels

# Load original label encoder (from Phase 3)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)  # original y before encoding

def evaluate_model(model, X_test, y_test, model_name):
    # Predict
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Decode labels
    y_pred_labels = encoder.inverse_transform(y_pred)
    y_true_labels = encoder.inverse_transform(y_true)

    # Classification report
    print(f"\n {model_name} - Classification Report:")
    print(classification_report(y_true_labels, y_pred_labels))

    # Confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title(f'{model_name} – Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return y_true, y_pred, y_true_labels, y_pred_labels, y_pred_probs


# In[68]:


# 5.3: Evaluate All Models


# In[70]:


cnn_results = evaluate_model(cnn_model, X_test_cnn, y_test, "CNN")


# In[72]:


lstm_results = evaluate_model(lstm_model, X_test_lstm, y_test, "LSTM")


# In[74]:


clstm_results = evaluate_model(clstm_model, X_test_clstm, y_test, "CLSTM")


# In[76]:


# 5.4: Extract 3 Correct + 3 Incorrect Predictions

def get_examples(X_test, y_true, y_pred, y_probs, label_names, num_correct=3, num_incorrect=3):
    correct_idxs = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
    incorrect_idxs = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]

    correct_examples = correct_idxs[:num_correct]
    incorrect_examples = incorrect_idxs[:num_incorrect]

    examples = []

    for i in correct_examples:
        examples.append({
            'type': 'correct',
            'index': i,
            'true': label_names[y_true[i]],
            'predicted': label_names[y_pred[i]],
            'confidence': np.max(y_probs[i])
        })

    for i in incorrect_examples:
        examples.append({
            'type': 'incorrect',
            'index': i,
            'true': label_names[y_true[i]],
            'predicted': label_names[y_pred[i]],
            'confidence': np.max(y_probs[i])
        })

    return examples


# In[86]:


## Final step Get Audio File Paths from Test Set"
# 1. Load ravdess_metadata.csv and encode the label column

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df_meta = pd.read_csv("ravdess_metadata.csv")

# Encode emotion labels numerically for stratification
le = LabelEncoder()
df_meta['label_encoded'] = le.fit_transform(df_meta['emotion'])


# In[88]:


# 2. Perform consistent split on the full metadata DataFrame

# 70% train, 15% val, 15% test
df_train, df_temp = train_test_split(df_meta, test_size=0.30, stratify=df_meta['label_encoded'], random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.50, stratify=df_temp['label_encoded'], random_state=42)


# In[96]:


# 3. Map test audio to the correct/incorrect predictions

# unpack them
y_true_cnn, y_pred_cnn, y_true_labels_cnn, y_pred_labels_cnn, probs_cnn = cnn_results


# In[98]:


correct_samples = []
incorrect_samples = []

for i in range(len(y_true_cnn)):
    if len(correct_samples) < 3 and y_true_cnn[i] == y_pred_cnn[i]:
        correct_samples.append(i)
    elif len(incorrect_samples) < 3 and y_true_cnn[i] != y_pred_cnn[i]:
        incorrect_samples.append(i)
    if len(correct_samples) == 3 and len(incorrect_samples) == 3:
        break


# In[100]:


# 4. Copy Audio Files to /samples/

import os
import shutil

os.makedirs("samples", exist_ok=True)

def copy_samples(indices, prefix):
    for i, idx in enumerate(indices):
        src_path = test_file_paths[idx]
        dst_path = f"samples/{prefix}_{i+1}.wav"
        shutil.copy(src_path, dst_path)
        print(f"Saved: {dst_path}")

copy_samples(correct_samples, "correct")
copy_samples(incorrect_samples, "incorrect")


# In[ ]:




