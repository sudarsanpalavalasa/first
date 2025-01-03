import streamlit as st
import cv2
import requests
import os
import time
import numpy as np

# Set up the upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
HEADERS = {"Authorization": "Bearer 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"}  # Replace with your API key

# Function to check if the model is ready
def check_model_ready():
    test_image = cv2.imencode(".jpg", np.zeros((100, 100, 3), dtype=np.uint8))[1].tobytes()
    response = requests.post(API_URL, headers=HEADERS, data=test_image)
    if response.status_code == 503:
        st.warning(f"Model is loading. Estimated time: {response.json().get('estimated_time', 'unknown')} seconds.")
        time.sleep(30)
        return check_model_ready()
    elif response.status_code == 200:
        st.success("Model is ready!")
    else:
        st.error(f"API error: {response.status_code} - {response.text}")

# Function to query the Hugging Face API with retries
def query_with_retries(image_data, retries=5, wait_time=20):
    for attempt in range(retries):
        response = requests.post(API_URL, headers=HEADERS, data=image_data)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            st.warning(f"Model is loading. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retries})")
            time.sleep(wait_time)
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
            return None
    st.error("Max retries reached. Could not process the frame.")
    return None

# Function to process video and detect objects
def process_video(video_file):
    video_path = os.path.join(UPLOAD_FOLDER, video_file.name)
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())

    # Display the uploaded video
    st.video(video_file)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_skip = 10  # Process every 10th frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip this frame

        st.write(f"Processing Frame {frame_count}...")

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, jpeg_frame = cv2.imencode(".jpg", frame_rgb)

        if ret:
            image_data = jpeg_frame.tobytes()
            output = query_with_retries(image_data)

            if output:
                st.write(f"API Response for Frame {frame_count}: {output}")  # Debugging output
                if 'boxes' in output:
                    for box in output['boxes']:
                        x1, y1, x2, y2 = map(int, box)
                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_count}", use_column_width=True)
            else:
                st.warning(f"No valid data returned for Frame {frame_count}")

    cap.release()

# Streamlit app
def main():
    st.header("Video Object Detection")

    upload_file = st.file_uploader("Choose a video...", type=["mp4"])

    if upload_file is not None:
        check_model_ready()
        process_video(upload_file)

if __name__ == "__main__":
    main()
