import streamlit as st

# app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")  # or your custom .pt

st.title("Bird Detector 🐦")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file:
    img = cv2.imdecode(
        np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR
    )
    results = model(img)
    annotated_frame = results[0].plot()
    st.image(annotated_frame, channels="BGR")
