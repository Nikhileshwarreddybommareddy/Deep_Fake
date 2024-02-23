import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# List of available models
model_names = ["InceptionV3", "DenseNet201", "Xception", "EfficientNetB4", "NASNetLarge", "VGG19", "NASNetMobile", "VGG16"]

# Function to extract frames from the video
def extract_frames(video_path, size=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frame = img_to_array(frame)
        frame = np.expand_dims(frame, axis=0)
        frames.append(frame)
    cap.release()
    return np.vstack(frames)

# Function to predict video using the selected model
def predict_video(video_path, model):
    frames = extract_frames(video_path)
    # Assuming that your model was trained with normalized images
    frames = frames.astype('float32') / 255.0
    predictions = model.predict(frames)
    avg_prediction = np.mean(predictions)
    return 'FAKE' if avg_prediction > 0.5 else 'REAL'

# Streamlit App
def main():
    st.title("Fake Video Detection")

    # Upload video
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

    if uploaded_file is not None:
        # Display uploaded video
        st.video(uploaded_file)

        # Model selection dropdown
        selected_model = st.selectbox("Select a model", model_names)

        # Load the selected model
        model = load_model(f"{selected_model}_transfer_learning_model.h5")

        # Button to make predictions
        if st.button("Make Prediction"):
            # Temporary file to save the uploaded video
            temp_file = f"temp_video.{uploaded_file.name.split('.')[-1]}"
            with open(temp_file, 'wb') as f:
                f.write(uploaded_file.read())

            # Predict video
            video_label = predict_video(temp_file, model)
            st.success(f"The video is predicted as: {video_label}")

            # Remove temporary file
            st.balloons()
            st.info("Prediction Complete!")
            st.info("Thank you for using our service!")

if __name__ == "__main__":
    main()
