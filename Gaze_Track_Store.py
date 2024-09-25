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

# Loading the model
model = load_model('gaze_tracking_gray_cnn50_01_.h5')
pd_pt = []

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Indices for the eye landmarks for blink
eye_indices = [33, 160, 158, 133, 153, 144]

# Define the right eye landmarks indices
RIGHT_EYE = [226, 113, 225, 224, 223, 222, 221, 190, 243, 112, 26, 22, 23, 24, 110, 25]

# Define screen dimensions
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Number of divisions
rows, cols = 4, 4

# Calculate the width and height of each grid area
cell_width = screen_width // cols
cell_height = screen_height // rows

###
##
#
# EAR threshold to detect a blink
EAR_THRESHOLD = 0.2
BLINK_CONSEC_FRAMES = 3

# Initialize the blink counter and blink times
blink_counter = 0
frame_counter = 0
blink_times = []

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the Euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize EAR data
ears = []
timestamps = []
###
##
#

# Function to calculate the center points of each grid area
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

# Calculate the center points
center_points = calculate_center_points(rows, cols, cell_width, cell_height)

# Create a blank screen
base_screen = cv2.imread('Nike_AD.png')
if base_screen is None:
    # If the image is not found, create a blank screen
    base_screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
else:
    base_screen = cv2.resize(base_screen, (screen_width, screen_height))

# Function to draw circle at random coordinate within radius from center point
def draw_circle(rounded_coordinate, center_points):
    x, y = rounded_coordinate
    if 0 <= x < rows and 0 <= y < cols:
        center_point = center_points[x][y]

        # Random angle and distance within the circle's radius
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, 200)

        # Compute new coordinates
        random_x = int(center_point[0] + distance * math.cos(angle))
        random_y = int(center_point[1] + distance * math.sin(angle))

        # Ensure coordinates are within screen boundaries
        random_x = max(0, min(screen_width - 1, random_x))
        random_y = max(0, min(screen_height - 1, random_y))

        # Draw circle
        screen = base_screen.copy()  # Reset screen to base
        draw_grid(screen)  # Draw grid on the reset screen
        time.sleep(0.0005)
        cv2.circle(screen, (random_x, random_y), radius=75, color=(0, 0, 255), thickness=5)

        # Store gaze point
        gaze = (random_x, random_y)
        pd_pt.append(gaze)

        return screen

# Draw the grid lines
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
    blink = frame
    # Convert the image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find the face landmarks
    results = face_mesh.process(rgb_frame)

    screen = base_screen.copy()  # Reset screen to base before processing each frame
    draw_grid(screen)  # Draw grid on the reset screen
    current_time = time.time() - start_time
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract the left eye region
            h, w, _ = frame.shape
            eye_region_points = np.array([
                (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
                for i in RIGHT_EYE
            ])

            # Calculate the bounding rectangle for the left eye region
            x, y, w, h = cv2.boundingRect(eye_region_points)
            # Crop the left eye region from the frame
            eye_region_image = frame[y:y + h, x:x + w]
            eye_region_image = cv2.flip(eye_region_image, 1)

            # Resize the cropped eye region image to 256x256
            eye_region_image = cv2.resize(eye_region_image, (256, 256))
            eye_region_image = cv2.cvtColor(eye_region_image, cv2.COLOR_BGR2GRAY)
            eye_region_image = eye_region_image / 255.0
            eye_region_image = np.array(eye_region_image)
            eye_region_image = np.expand_dims(eye_region_image, axis=-1)  # Add channel dimension
            eye_region_image = np.expand_dims(eye_region_image, axis=0)  # Add batch dimension
            # Now eye_region_image shape is (1, 256, 256, 1)

            # Predict the coordinate
            predicted_coordinate = abs(model.predict(eye_region_image))

            coordinate = predicted_coordinate[0]

            # Round each element of the coordinate
            rounded_coordinate = (round(coordinate[0]), round(coordinate[1]))

            # Draw the circle and get the updated screen
            screen = draw_circle(rounded_coordinate, center_points)

            left_eye = []
            for idx in eye_indices:
                x = int(face_landmarks.landmark[idx].x * blink.shape[1])
                y = int(face_landmarks.landmark[idx].y * blink.shape[0])
                left_eye.append((x, y))


            # Calculate EAR for the eye
            ear = eye_aspect_ratio(left_eye)

            # Append EAR and timestamp
            current_time = time.time() - start_time
            ears.append(ear)
            timestamps.append(current_time)

            # Check if EAR is below the blink threshold
            if ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= BLINK_CONSEC_FRAMES:
                    blink_counter += 1
                    blink_times.append(current_time)
                frame_counter = 0
           # current_time = time.time() - start_time

    # Display the result in full screen mode
    if screen is not None and screen.size > 0:
        cv2.namedWindow('Grid', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Grid', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Grid', screen)

    if current_time >= 15:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Flatten the coordinates into a single column format
flattened_coordinates = [f"{x}, {y}" for x, y in pd_pt]

# Create DataFrame
df = pd.DataFrame(flattened_coordinates, columns=['Coordinate'])

# Define the file name
file_name = 'gaze.xlsx'

# Remove the existing file if it exists
if os.path.exists(file_name):
    os.remove(file_name)

# Save DataFrame to Excel
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
FileName= 'Blink.xlsx'
# Remove the existing file if it exists
if os.path.exists(FileName):
    os.remove(FileName)

# Save DataFrame to Excel
df1.to_excel(FileName, index=False, engine='openpyxl')