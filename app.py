import streamlit as st
import av
import cv2
import pygame
import requests
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO
import numpy as np

# Load YOLO pre-trained model
model = YOLO('yolo11n.pt')  # Ensure the correct model path

# Paths to audio files for locations
audio_files = {
    "left": "/Users/hadeel/Desktop/SADIA/left.mp3",
    "center": "/Users/hadeel/Desktop/SADIA/center.mp3",
    "right": "/Users/hadeel/Desktop/SADIA/right.mp3"
}

# Initialize pygame mixer
pygame.mixer.init()

# Setup Streamlit interface
st.title("Real-Time Object Detection with YOLO and Audio Feedback")

# Store whether the webcam stream is active or not
if 'video_active' not in st.session_state:
    st.session_state.video_active = False  # Default to not active

# Function to send audio request to API for object name
def get_object_audio(object_name):
    # Send a request to the API to get the TTS audio for the object
    response = requests.post(
        "https://api.kokorotts.com/v1/audio/speech",  # Example Kokoro TTS API endpoint
        json={
            "model": "kokoro",  # Not used but required for compatibility
            "input": f"It's a {object_name}",
            "voice": "af_bella",  # You can change the voice here
            "response_format": "mp3",  # Supported: mp3, wav, opus, flac
            "speed": 1.0
        }
    )
    
    if response.status_code == 200:
        # Return the audio content in bytes
        return response.content  
    else:
        st.error("Failed to get audio from API")
        return None

# Function to play audio files
def play_audio(files):
    for file in files:
        pygame.mixer.music.load(file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Wait until audio finishes playing
            continue

# Function to process frames for object detection
def process_frame(frame):
    H, W, _ = frame.shape  # Get frame dimensions
    results = model(frame)  # Perform detection with YOLO model
    
    # Process the detection results
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

            # Determine the position of the object in the frame (left, center, right)
            if centerX <= W / 3:
                W_pos = "left"
            elif centerX <= (W / 3 * 2):
                W_pos = "center"
            else:
                W_pos = "right"

            # Construct position description
            position = f"The {class_detected_name} is at the {W_pos}"
            st.write(position)
            
            # Play the corresponding audio files for the location
            play_audio([audio_files[W_pos]])

            # Get the TTS audio for the detected object and play it
            object_audio = get_object_audio(class_detected_name)
            if object_audio:
                with open(f"{class_detected_name}_audio.mp3", "wb") as f:
                    f.write(object_audio)
                pygame.mixer.music.load(f"{class_detected_name}_audio.mp3")
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():  # Wait until the object audio finishes
                    continue

            # Add text label with a background rectangle
            text = f'{class_detected_name} ({confidence:.2f}%)'
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), (0, 0, 255), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

# Callback for Streamlit WebRTC to handle incoming video frames
def video_frame_callback(frame):
    # Convert the frame to numpy array for processing
    array = process_frame(frame.to_ndarray(format="bgr24"))

    # Return the processed frame for Streamlit to display
    return av.VideoFrame.from_ndarray(array, format="bgr24")

# Button to start/stop the webcam stream
if st.button("Start/Stop Webcam Stream"):
    st.session_state.video_active = not st.session_state.video_active

# If the webcam stream is active, start the WebRTC streamer
if st.session_state.video_active:
    webrtc_streamer(
        key="yolo-webrtc",
        video_frame_callback=video_frame_callback,
        mode=WebRtcMode.SENDRECV,  # Sends and receives video stream
        media_stream_constraints={
            "video": True,
            "audio": False  # Set to False if you don't need audio
        }
    )
else:
    st.write("Click the button to start the webcam stream.")
