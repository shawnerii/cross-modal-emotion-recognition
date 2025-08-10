import cv2
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("models/fer_vgg_best.h5")

# Define class labels in the correct order
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Prediction function
def predict_emotion(image):
    if image is None:
        return "No image uploaded"

    # Convert to grayscale and resize to match model input
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized.astype("float32") / 255.0
    reshaped = np.expand_dims(normalized, axis=(0, -1))  # (1, 48, 48, 1)

    # Predict
    prediction = model.predict(reshaped)[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index] * 100
    predicted_label = class_names[predicted_index]

    return f"{predicted_label} ({confidence:.2f}%)"

# Gradio interface (no shape argument)
iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(label="Upload a face image", image_mode="RGB"),
    outputs=gr.Textbox(label="Predicted Emotion"),
    title="FER2013 Emotion Classifier",
    description="Upload a facial image (RGB) and get predicted emotion based on FER2013-trained CNN."
)

if __name__ == "__main__":
    iface.launch()