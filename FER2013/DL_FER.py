#!/usr/bin/env python
# coding: utf-8

# In[15]:


## PHASE 1 – DATA PREPARATION ##
# Step 1: Import Required Libraries

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


# In[17]:


# Step 2: Define Paths and Image Settings

train_dir = "archive/train"
test_dir = "archive/test"

# Common settings
img_height = 48
img_width = 48
batch_size = 32


# In[19]:


# Step 3: Create Image Generators (with validation split)

# Train: 85% train / 15% validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation',
    shuffle=True
)


# In[21]:


# Step 4: Test Generator for Final Evaluation

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)


# In[25]:


## PHASE 2: Model Building (VGGNet for FER2013) ##
# Step 1: Import Required Libraries


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization)


# In[27]:


# Step 2: Define the VGG-style CNN Model

def build_vgg_fer(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # FC Layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    return model


# In[29]:


# Step 3: Compile the Model

model = build_vgg_fer()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# In[31]:


## Phase 3: Model Construction – VGG-Style CNN for Facial Emotion Recognition (FER2013) ##
# 1. Import Required Libraries

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


# In[33]:


# 2. Define the VGG-style CNN Model

def build_vgg_style_cnn(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and Dense layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    return model


# In[35]:


# 3. Compile the Model

model = build_vgg_style_cnn()

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# In[37]:


## PHASE 4: Model Training (FER2013 VGGNet) ##
# Step 1: Import Callbacks

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os


# In[39]:


# Step 2: Define Callbacks

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    ModelCheckpoint('models/fer_vgg_best.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
]


# In[41]:


os.makedirs("models", exist_ok=True)


# In[43]:


# Step 3: Fit the Model

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)


# In[45]:


# Step 4: Plot Accuracy and Loss

import matplotlib.pyplot as plt

def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(14,5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Val Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


# In[47]:


plot_training(history)


# In[49]:


## PHASE 5: Model Evaluation ##
# Step 1: Load Best Model & Test Data

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load best model
model = load_model("models/fer_vgg_best.h5")

# Predict on test set
test_generator.reset()  # reset generator index
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())


# In[51]:


# Step 2: Classification Report & Confusion Matrix

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Report
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[53]:


# Step 3: Extract 3 Correct + 3 Incorrect Images

import shutil
import os

# Get file paths
filenames = test_generator.filepaths
correct = []
incorrect = []

for i, (true, pred) in enumerate(zip(y_true, y_pred)):
    if true == pred and len(correct) < 3:
        correct.append(i)
    elif true != pred and len(incorrect) < 3:
        incorrect.append(i)
    if len(correct) == 3 and len(incorrect) == 3:
        break


# In[55]:


os.makedirs("samples", exist_ok=True)

def copy_images(indices, label):
    for i, idx in enumerate(indices):
        src = filenames[idx]
        dst = f"samples/{label}_{i+1}.png"
        shutil.copy(src, dst)
        print(f"Saved: {dst}")

copy_images(correct, "correct")
copy_images(incorrect, "incorrect")


# In[ ]:




