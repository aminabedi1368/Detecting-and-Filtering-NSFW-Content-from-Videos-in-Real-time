# app/main.py

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from model import load_model, predict_frame
from utils import preprocess_frame

# Defining directories
data_dir = 'video_frames'
safe_dir = os.path.join(data_dir, 'safe')
nsfw_dir = os.path.join(data_dir, 'nsfw')

# Function to extract frames using OpenCV
def extract_frames(video_path, output_dir, frame_rate=1):
    os.makedirs(output_dir, exist_ok=True)
    video_capture = cv2.VideoCapture(video_path)
    success, frame = video_capture.read()
    count = 0
    while success:
        if count % frame_rate == 0:
            frame_filename = os.path.join(output_dir, f'frame_{count}.jpg')
            cv2.imwrite(frame_filename, frame)
        success, frame = video_capture.read()
        print('count: ', count)
        count += 1
    video_capture.release()

# Extract all frames from every video in all subdirectories in a directory
def process_videos():
    for video in os.listdir(safe_dir):
        extract_frames(os.path.join(safe_dir, video), 'frames/safe')
    for video in os.listdir(nsfw_dir):
        extract_frames(os.path.join(nsfw_dir, video), 'frames/nsfw')

# Load the trained model
model = load_model('../models/cnn_lstm_model.h5')

# Define RTMP stream URL (example URL, adjust as needed)
stream_url = 'rtmp://your.stream.url/live'

# Open the RTMP stream
cap = cv2.VideoCapture(stream_url)

# Process videos to extract frames
process_videos()

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Prepare data generators
train_generator = datagen.flow_from_directory(
    'frames/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
validation_generator = datagen.flow_from_directory(
    'frames/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Predict using the model
    prediction = predict_frame(model, preprocessed_frame)
    print(prediction)

    # Display the frame
    cv2.imshow('RTMP Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Class distribution plot
class_distribution = {'Safe': len(os.listdir('frames/safe')), 'NSFW': len(os.listdir('frames/nsfw'))}
plt.figure(figsize=(8,6))
plt.bar(class_distribution.keys(), class_distribution.values(), color=['blue', 'red'])
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig('class_distribution.png')
plt.show()

# Function to calculate pixel intensities
def calculate_pixel_intensities(frame_dir):
    pixel_intensities = []
    for frame in os.listdir(frame_dir):
        img = cv2.imread(os.path.join(frame_dir, frame), cv2.IMREAD_GRAYSCALE)
        pixel_intensities.extend(img.flatten())
    return pixel_intensities

# Calculate pixel intensities for safe and NSFW frames
safe_pixel_intensity = calculate_pixel_intensities('frames/safe')
nsfw_pixel_intensity = calculate_pixel_intensities('frames/nsfw')

# Plotting histograms of pixel intensities
plt.figure(figsize=(12, 6))
plt.hist(safe_pixel_intensity, bins=50, color='blue', alpha=0.5, label='Safe Frames')
plt.hist(nsfw_pixel_intensity, bins=50, color='red', alpha=0.5, label='NSFW Frames')
plt.title('Pixel Intensity Distribution')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('pixel_intensity_distribution.png')
plt.show()

# Load pre-trained VGG16 model for heatmap generation
vgg_model = VGG16(weights='imagenet')
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Function to generate heatmap
def generate_heatmap(frame_path):
    img = image.load_img(frame_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = vgg_model.predict(x)
    top_class_index = np.argmax(preds[0])
    output = vgg_model.output[:, top_class_index]
    last_conv_layer = vgg_model.get_layer("block5_conv3")
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([vgg_model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

# Generate and visualize heatmaps for a sample of NSFW frames
plt.figure(figsize=(20, 15))
for i, frame_path in enumerate(os.listdir('frames/nsfw')[:10]):
    heatmap = generate_heatmap(os.path.join('frames/nsfw', frame_path))
    plt.subplot(3, 4, i+1)
    plt.imshow(heatmap)
    plt.title(f'Frame {i+1}')
    plt.axis('off')
plt.savefig('heatmaps.png')
plt.show()
