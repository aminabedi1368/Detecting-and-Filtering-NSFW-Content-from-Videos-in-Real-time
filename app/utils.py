# app/utils.py
import cv2
import numpy as np

def preprocess_frame(frame):
    # Resize frame to 64x64
    frame = cv2.resize(frame, (64, 64))
    # Normalize pixel values
    frame = frame / 255.0
    return frame
