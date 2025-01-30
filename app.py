import streamlit as st
import av
import cv2
import pygame
import requests
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API URL from environment variables
api_url = os.getenv('API_URL', 'https://api.kokorotts.com/v1/audio/speech')  # Use default URL if not set

# Load YOLO pre-trained model
model = YOLO('yolo11n.pt')

# Initialize pygame mixer
pygame.mixer.init()

# Setup Streamlit interface
st.title("Real-Time Object Detection with YOLO and Audio Feedback")

# Store whether the webcam stream is active or not
if 'video_active' not in st.session_state:
    st.session_state.video_active = False  # Default to not active

# Function to send audio request to API for object name
def get_object_audio(object_name):
    response = requests.post(
        api_url,
        json={
            "model": "kokoro",
            "input": f"It's a {object_name}",
            "voice": "af_bella",
            "response_format": "mp3",
            "speed": 1.0
        }
    )
    
    if response.status_code == 200:
        return response.content
    else:
        st.error("Failed to get audio from API")
        return None

# Function to play audio files
def play_audio(files):
    for file in files:
        pygame.mixer.music.load(file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue

# Function to process frames for object detection
def process_frame(frame):
    H, W, _ = frame.shape
    results = model(frame)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype('int')
            confidence = float(box.conf[0].numpy())
            class_detected_number = int(box.cls[0])
            class_detected_name = result.names[class_detected_number]

            # Draw bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Calculate the center of the box
            centerX = (x1 + x2) // 2
            W_pos = "left" if centerX <= W / 3 else "center" if centerX <= (W / 3 * 2) else "right"

            # Construct position description
            position = f"The {class_detected_name} is at the {W_pos}"
            st.write(position)

            # Play corresponding location audio
            audio_files = {
                "left": "left.mp3",  # Example path, consider changing to relative paths or dynamic loading
                "center": "center.mp3",
                "right": "right.mp3"
            }
            play_audio([audio_files[W_pos]])

            # Get object audio and play it
            object_audio = get_object_audio(class_detected_name)
            if object_audio:
                with open(f"{class_detected_name}_audio.mp3", "wb") as f:
                    f.write(object_audio)
                pygame.mixer.music.load(f"{class_detected_name}_audio.mp3")
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    continue

            # Add text label
            text = f'{class_detected_name} ({confidence:.2f}%)'
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), (0, 0, 255), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

# Callback for Streamlit WebRTC to handle incoming video frames
def video_frame_callback(frame):
    array = process_frame(frame.to_ndarray(format="bgr24"))
    return av.VideoFrame.from_ndarray(array, format="bgr24")

# Button to start/stop the webcam stream
if st.button("Start/Stop Webcam Stream"):
    st.session_state.video_active = not st.session_state.video_active

# If the webcam stream is active, start the WebRTC streamer
if st.session_state.video_active:
    webrtc_streamer(
        key="yolo-webrtc",
        video_frame_callback=video_frame_callback,
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )
else:
    st.write("Click the button to start the webcam stream.")
