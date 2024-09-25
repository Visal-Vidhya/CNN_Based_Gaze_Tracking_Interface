import cv2
import numpy as np
import mediapipe as mp
from screeninfo import get_monitors
import time
import os
import string
import random

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define the right eye landmarks indices

RIGHT_EYE = [226, 113, 225, 224, 223, 222, 221, 190, 243, 112, 26, 22, 23, 24, 110, 25]

#N = 7

def draw_grid(image):
    # Get screen resolution
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height

    rows = 4
    cols = 4
    cell_width = screen_width // cols
    cell_height = screen_height // rows

    # Draw the grid lines
    for i in range(rows + 1):
        cv2.line(image, (0, i * cell_height), (screen_width, i * cell_height), (255, 255, 255), 2)
    for j in range(cols + 1):
        cv2.line(image, (j * cell_width, 0), (j * cell_width, screen_height), (255, 255, 255), 2)


def pulsate_red_point(image, center_x, center_y, duration, capture_eye_images, save_dir):
    num_frames = 30  # Number of frames for pulsation
    start_time = time.time()
    max_radius = 20  # Maximum radius of the pulsating point
    min_radius = 5  # Minimum radius of the pulsating point
    frame_count = 0  # Count the frames for naming saved images

    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time
        phase = (elapsed_time / duration) % 1  # Phase of pulsation (0 to 1)
        radius = int(min_radius + (max_radius - min_radius) * abs(0.5 - phase) * 2)  # Pulsate effect

        # Clear the image
        image[:] = 0
        # Draw the grid lines
        draw_grid(image)
        # Draw the pulsating red point
        cv2.circle(image, (center_x, center_y), radius, (0, 0, 255), -1)
        # Display the image
        cv2.imshow('Grid', image)

        if cv2.getWindowProperty('Grid', cv2.WND_PROP_VISIBLE) < 1:
            return False  # If the 'Grid' window is closed, return False

        if cv2.waitKey(int(duration * 1000 / num_frames)) & 0xFF == ord('q'):
            return False  # If 'q' is pressed, return False

        # Capture and save right eye images
        if capture_eye_images:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame to find the face landmarks
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract the right eye region
                    h, w, _ = frame.shape
                    right_eye_points = np.array(
                        [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in
                         RIGHT_EYE])

                    # Calculate the bounding rectangle for the right eye region
                    x, y, w, h = cv2.boundingRect(right_eye_points)

                    # Crop the right eye from the frame
                    right_eye_image = frame[y:y + h, x:x + w]
                    right_eye_image = cv2.flip(right_eye_image, 1)
                    right_eye_image = cv2.cvtColor(right_eye_image, cv2.COLOR_BGR2GRAY)
                    right_eye_image = cv2.resize(right_eye_image, (256, 256))

                    # Save the right eye image
                    if right_eye_image is not None:
                        #res = ''.join(random.choices(string.ascii_letters, k=N))
                        image_path = os.path.join(save_dir, f"A_GridA_{frame_count:04d}.png")
                        cv2.imwrite(image_path, right_eye_image)
                        frame_count += 1

    return True  # Continue animation


def create_grid_with_animation(capture_eye_images=False):
    # Get the screen resolution
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height

    rows = 4
    cols = 4
    cell_width = screen_width // cols
    cell_height = screen_height // rows

    # Create a black image with the same dimensions as the screen
    image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # Display the image in full screen mode
    cv2.namedWindow('Grid', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Grid', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    total_duration = 5 * rows * cols  # Total duration of the animation
    start_time = time.time()

    while time.time() - start_time < total_duration:
        for i in range(rows):
            for j in range(cols):
                cell_start_x = j * cell_width
                cell_start_y = i * cell_height
                cell_end_x = (j + 1) * cell_width
                cell_end_y = (i + 1) * cell_height
                center_x = (cell_start_x + cell_end_x) // 2
                center_y = (cell_start_y + cell_end_y) // 2

                # Create directory to save right eye images
                grid_label = f"grid_{i}_{j}"
                save_dir = os.path.join("eye_gray", grid_label)
                os.makedirs(save_dir, exist_ok=True)

                # Animate the pulsating point within the cell for 3 seconds
                if not pulsate_red_point(image, center_x, center_y, 5, capture_eye_images, save_dir):
                    return  # Exit the function if animation is stopped


def main():
    global cap
    # Initialize video capture
    cap = cv2.VideoCapture(1)

    # Create a black image for the grid animation
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height
    grid_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    grid_animation_thread = threading.Thread(target=create_grid_with_animation, args=(True,))
    grid_animation_thread.start()

    right_eye_image = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find the face landmarks
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract the right eye region
                h, w, _ = frame.shape
                right_eye_points = np.array(
                    [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE])

                # Calculate the bounding rectangle for the right eye region
                x, y, w, h = cv2.boundingRect(right_eye_points)

                # Crop the right eye from the frame
                right_eye_image = frame[y:y + h, x:x + w]
                right_eye_image = cv2.flip(right_eye_image, 1)
                right_eye_image = cv2.resize(right_eye_image, (256, 256))

        if right_eye_image is not None:
            # Display the right eye
            cv2.imshow('Right Eye', right_eye_image)

        if cv2.getWindowProperty('Grid', cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty('Right Eye',
                                                                                            cv2.WND_PROP_VISIBLE) < 1:
            break  # Exit if any of the windows are closed

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit if 'q' is pressed

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import threading

    main()
