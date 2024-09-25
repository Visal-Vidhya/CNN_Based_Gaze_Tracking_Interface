import math
import os
import cv2
import numpy as np
import random
from screeninfo import get_monitors
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import pandas as pd
import time
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt

# Load the model
model = load_model('gaze_tracking_gray_cnn50_1.h5')
pd_pt = []

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Eye landmarks indices
eye_indices = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [226, 113, 225, 224, 223, 222, 221, 190, 243, 112, 26, 22, 23, 24, 110, 25]

# Screen dimensions
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Grid setup
rows, cols = 4, 4
cell_width = screen_width // cols
cell_height = screen_height // rows

# EAR threshold and counters
EAR_THRESHOLD = 0.2
BLINK_CONSEC_FRAMES = 3
blink_counter = 0
frame_counter = 0
blink_times = []


# EAR calculation function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Initialize EAR data
ears = []
timestamps = []


# Calculate center points of grid cells
def calculate_center_points(rows, cols, cell_width, cell_height):
    centers = []
    for i in range(rows):
        row_centers = []
        for j in range(cols):
            center_x = j * cell_width + cell_width // 2
            center_y = i * cell_height + cell_height // 2
            row_centers.append((center_x, center_y))
        centers.append(row_centers)
    return centers


center_points = calculate_center_points(rows, cols, cell_width, cell_height)

# Open the video file
video_path = 'Nike_VAd.mp4'
video_capture = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not video_capture.isOpened():
    raise ValueError(f"Error opening video file {video_path}")


# Function to draw a circle at a random position within a radius from the center point
def draw_circle(rounded_coordinate, center_points):
    x, y = rounded_coordinate
    if 0 <= x < rows and 0 <= y < cols:
        center_point = center_points[x][y]
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, 200)
        random_x = int(center_point[0] + distance * math.cos(angle))
        random_y = int(center_point[1] + distance * math.sin(angle))
        random_x = max(0, min(screen_width - 1, random_x))
        random_y = max(0, min(screen_height - 1, random_y))
        screen = base_screen.copy()
        draw_grid(screen)
        time.sleep(0.0005)
        cv2.circle(screen, (random_x, random_y), radius=75, color=(0, 0, 255), thickness=5)
        pd_pt.append((random_x, random_y))
        return screen


# Draw grid lines on the screen
def draw_grid(screen):
    for i in range(rows + 1):
        cv2.line(screen, (0, i * cell_height), (screen_width, i * cell_height), (255, 255, 255), 2)
    for j in range(cols + 1):
        cv2.line(screen, (j * cell_width, 0), (j * cell_width, screen_height), (255, 255, 255), 2)


# Start video capture
cap = cv2.VideoCapture(0)
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ret_video, base_screen = video_capture.read()
    if not ret_video:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_video, base_screen = video_capture.read()
    base_screen = cv2.resize(base_screen, (screen_width, screen_height))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    screen = base_screen.copy()
    draw_grid(screen)
    current_time = time.time() - start_time
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            eye_region_points = np.array(
                [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE])
            x, y, w, h = cv2.boundingRect(eye_region_points)
            eye_region_image = frame[y:y + h, x:x + w]
            eye_region_image = cv2.flip(eye_region_image, 1)
            eye_region_image = cv2.resize(eye_region_image, (256, 256))
            eye_region_image = cv2.cvtColor(eye_region_image, cv2.COLOR_BGR2GRAY)
            eye_region_image = eye_region_image / 255.0
            eye_region_image = np.expand_dims(eye_region_image, axis=[-1, 0])

            predicted_coordinate = abs(model.predict(eye_region_image))
            rounded_coordinate = (round(predicted_coordinate[0][0]), round(predicted_coordinate[0][1]))
            screen = draw_circle(rounded_coordinate, center_points)

            left_eye = [(int(face_landmarks.landmark[idx].x * frame.shape[1]),
                         int(face_landmarks.landmark[idx].y * frame.shape[0])) for idx in eye_indices]
            ear = eye_aspect_ratio(left_eye)
            current_time = time.time() - start_time
            ears.append(ear)
            timestamps.append(current_time)

            if ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= BLINK_CONSEC_FRAMES:
                    blink_counter += 1
                    blink_times.append(current_time)
                frame_counter = 0

    if screen is not None and screen.size > 0:
        cv2.namedWindow('Grid', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Grid', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Grid', screen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if current_time >= 15:
        break
cap.release()
cv2.destroyAllWindows()

# Save gaze data to Excel
flattened_coordinates = [f"{x}, {y}" for x, y in pd_pt]
df = pd.DataFrame(flattened_coordinates, columns=['Coordinate'])
file_name = 'gaze_V.xlsx'
if os.path.exists(file_name):
    os.remove(file_name)
df.to_excel(file_name, index=False, engine='openpyxl')

# Ensure blink_times is the same length as timestamps for proper alignment
max_len = max(len(timestamps), len(blink_times))
timestamps.extend([None] * (max_len - len(timestamps)))
ears.extend([None] * (max_len - len(ears)))
blink_times.extend([None] * (max_len - len(blink_times)))

# Create a DataFrame for timestamps, ears, and blink_times
data = {
    'Timestamps': timestamps,
    'EAR': ears,
    'Blink Times': blink_times
}
df1 = pd.DataFrame(data)
FileName = 'Blink_V.xlsx'
if os.path.exists(FileName):
    os.remove(FileName)
df1.to_excel(FileName, index=False, engine='openpyxl')
