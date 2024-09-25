import cv2
import numpy as np
import math
import random
from screeninfo import get_monitors
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
import threading
import queue
import time

# Loading the model
model = load_model('gaze_tracking_gray_cnn50_01_.h5')
pd_pt = []

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define the right eye landmarks indices
RIGHT_EYE = [226, 113, 225, 224, 223, 222, 221, 190, 243, 112, 26, 22, 23, 24, 110, 25]

# Define screen dimensions
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Number of divisions
rows, cols = 4, 4
cell_width = screen_width // cols
cell_height = screen_height // rows


# Calculate Mean Absolute Deviation (MAD)
def calculate_mad(gaze_data, reference_trajectory):
    return np.mean(np.abs(gaze_data - reference_trajectory))


# Calculate Root Mean Square Error (RMSE)
def calculate_rmse(gaze_data, reference_trajectory):
    return np.sqrt(np.mean((gaze_data - reference_trajectory) ** 2))


# Calculate Accuracy
def calculate_accuracy(gaze_data, reference_trajectory, threshold):
    correct_points = 0
    total_points = len(gaze_data)

    for gaze_point in gaze_data:
        if np.any(np.linalg.norm(gaze_point - reference_trajectory, axis=1) <= threshold):
            correct_points += 1

    return correct_points / total_points


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


# Interpolation function
def interpolate(start, end, t):
    return start + (end - start) * t


# Define the path
positions = [(0, 0), (0, 3), (3, 3), (3, 0), (0, 0)]
steps = 200

# Calculate the ball's path
ball_positions = []
for k in range(len(positions) - 1):  # iterate upto second last element
    start_pos = positions[k]
    end_pos = positions[k + 1]

    # Calculate start and end coordinates
    start_x = start_pos[1] * cell_width + cell_width // 2
    start_y = start_pos[0] * cell_height + cell_height // 2
    end_x = end_pos[1] * cell_width + cell_width // 2
    end_y = end_pos[0] * cell_height + cell_height // 2

    for i in range(steps + 1):
        t = i / steps
        ball_x = interpolate(start_x, end_x, t)
        ball_y = interpolate(start_y, end_y, t)
        ball_positions.append((ball_x, ball_y))


# Function to draw the ball
def draw_ball(screen, x, y, radius=150, color=(255, 0, 0)):
    cv2.circle(screen, (int(x), int(y)), radius, color, -1)  # Draw solid circle


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
        cv2.circle(screen, (random_x, random_y), radius=50, color=(0, 255, 0), thickness=3)

        # Store gaze point
        gaze = (random_x, random_y)
        pd_pt.append(gaze)

        return screen


# Draw the grid lines
def draw_grid(screen):
    for i in range(rows + 1):
        cv2.line(screen, (0, i * cell_height), (screen_width, i * cell_height), (0, 0, 0), 2)
    for j in range(cols + 1):
        cv2.line(screen, (j * cell_width, 0), (j * cell_width, screen_height), (0, 0, 0), 2)


# Capture thread function
def capture_frames(cap, frame_queue, frame_skip):
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame_queue.put(frame)
        frame_count += 1


# Initialize video capture
cap = cv2.VideoCapture(0)

# Frame queue
frame_queue = queue.Queue()

# Frame skip parameter
frame_skip = 3

# Start capture thread
capture_thread = threading.Thread(target=capture_frames, args=(cap, frame_queue, frame_skip))
capture_thread.start()

# Animation index
index = 0

while cap.isOpened():
    if frame_queue.empty():
        time.sleep(0.01)  # Wait a bit if the queue is empty
        continue

    frame = frame_queue.get()
    if frame is None:
        break

    # Create a blank screen
    base_screen = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    if index < len(ball_positions):
        # Get the current position of the ball
        ball_x, ball_y = ball_positions[index]
        index += 1  # Move to the next position
    else:
        break

    # Draw the ball on the base screen
    draw_ball(base_screen, ball_x, ball_y)

    # Convert the image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find the face landmarks
    results = face_mesh.process(rgb_frame)

    screen = base_screen.copy()  # Reset screen to base before processing each frame
    draw_grid(screen)  # Draw grid on the reset screen

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
            predicted_coordinate = model.predict(eye_region_image)

            coordinate = predicted_coordinate[0]

            # Round each element of the coordinate
            rounded_coordinate = (round(coordinate[0]), round(coordinate[1]))

            # Draw the circle and get the updated screen
            screen = draw_circle(rounded_coordinate, center_points)

    # Display the result in full screen mode
    if screen is not None and screen.size > 0:
        cv2.namedWindow('Grid', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Grid', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Grid', screen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Convert to numpy arrays for calculations
pd_pt = np.array(pd_pt)
ball_positions = np.array(ball_positions)

# Ensure lengths are equal for comparison (you may need to handle this according to your data specifics)
min_len = min(len(pd_pt), len(ball_positions))
pd_pt = pd_pt[:min_len]
ball_positions = ball_positions[:min_len]

# Calculate metrics
mad = calculate_mad(pd_pt, ball_positions)
rmse = calculate_rmse(pd_pt, ball_positions)
dtw_distance, _ = fastdtw(pd_pt, ball_positions, dist=dist.euclidean)
threshold_distance = 150
accuracy = calculate_accuracy(pd_pt, ball_positions, threshold_distance)

# Visualization
# Plot results
plt.figure(figsize=(12, 8))

# Plot reference trajectory
plt.plot(ball_positions[:, 0], ball_positions[:, 1], label='Ball Trajectory', linestyle='--', marker='o')

# Plot gaze data
plt.plot(pd_pt[:, 0], pd_pt[:, 1], label='Gaze Data', linestyle='-', marker='x')

# Plot alignment paths
for (i, j) in zip(range(len(pd_pt)), range(len(ball_positions))):
    plt.plot([ball_positions[i][0], pd_pt[j][0]], [ball_positions[i][1], pd_pt[j][1]], 'k-', lw=0.5)

# Plot threshold circles
for point in ball_positions:
    circle = plt.Circle(point, threshold_distance, color='g', fill=False, linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'{accuracy * 100:.2f}% Trajectory Accuracy')
plt.grid(True)
plt.axis('equal')
plt.savefig('Trajectory Accuracy')
plt.show()

print(f"MAD: {mad}")
print(f"RMSE: {rmse}")
print(f"DTW Distance: {dtw_distance}")
print(f"Accuracy: {accuracy * 100:.2f}%")
