import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from screeninfo import get_monitors
import tkinter as tk
from tkinter import scrolledtext
from tabulate import tabulate

# Define screen dimensions
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Load gaze coordinates
G_coordinates = pd.read_excel('gaze_V.xlsx', engine='openpyxl')
coordinates = G_coordinates['Coordinate']

# Load blink data
blink_D = pd.read_excel('Blink_V.xlsx', engine='openpyxl')
timestamps = blink_D['Timestamps']
ears = blink_D['EAR']
blink_times = blink_D['Blink Times']
blink = blink_times.dropna()
blink_count = len(blink)

# Convert coordinates from string to tuples of integers
coordinates = [tuple(map(int, coord.split(', '))) for coord in coordinates]

# Define 4x4 grid regions
num_rows = 4
num_cols = 4
region_width = screen_width // num_cols
region_height = screen_height // num_rows

regions = {}
for row in range(num_rows):
    for col in range(num_cols):
        x1 = col * region_width
        y1 = row * region_height
        x2 = (col + 1) * region_width
        y2 = (row + 1) * region_height
        regions[f'Region {row * num_cols + col + 1}'] = (x1, y1, x2, y2)


# Function to draw numbered regions on the image
def draw_grid(image):
    for region_name, (x1, y1, x2, y2) in regions.items():
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        text_position = (x1 + 10, y1 + 30)
        cv2.putText(image, region_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


# Load the video
video_path = 'Nike_VAd.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError(f"Error opening video file {video_path}")

# Create a window for displaying the video
cv2.namedWindow('Gaze Tracking', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Gaze Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Initialize the heatmap
heatmap = np.zeros((screen_height, screen_width), dtype=np.float32)

coordinate_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and convert the frame to RGB
    frame = cv2.resize(frame, (screen_width, screen_height))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Add gaze coordinates to the heatmap
    if coordinate_index < len(coordinates):
        x, y = coordinates[coordinate_index]
        heatmap[y, x] += 1
        coordinate_index += 1

    # Apply Gaussian blur to smooth the heatmap
    heatmap_blurred = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=15, sigmaY=15)

    # Normalize the heatmap to the range [0, 1]
    heatmap_normalized = heatmap_blurred / heatmap_blurred.max()

    # Enhance contrast by scaling the heatmap values
    contrast_scale = 3
    heatmap_contrasted = np.clip(heatmap_normalized * contrast_scale, 0, 1)

    # Apply colormap
    colormap = plt.get_cmap('jet')
    heatmap_colored = colormap(heatmap_contrasted)
    heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)

    # Overlay the heatmap on the frame
    overlay = cv2.addWeighted(rgb_frame, 0.6, heatmap_colored, 0.4, 0)

    # Draw the grid
    draw_grid(overlay)

    # Convert the overlay back to BGR for OpenCV display
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    # Display the result
    cv2.imshow('Gaze Tracking', overlay_bgr)

    # Check if all coordinates have been processed
    if coordinate_index >= len(coordinates):
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Count the number of gaze points in each region
region_counts = {name: 0 for name in regions}

for (x, y) in coordinates:
    for region_name, (x1, y1, x2, y2) in regions.items():
        if x1 <= x < x2 and y1 <= y < y2:
            region_counts[region_name] += 1
            break

# Find the region with the highest gaze fixation
max_region = max(region_counts, key=region_counts.get)
max_count = region_counts[max_region]

print(f"Region with the highest gaze fixation: {max_region} ({max_count} gaze points)")

# Plot the gaze points distribution across regions using a histogram
plt.figure(figsize=(10, 6))
labels = list(region_counts.keys())
counts = list(region_counts.values())
colors = plt.cm.jet(np.linspace(0, 1, len(labels)))

plt.bar(labels, counts)
plt.title('Gaze Fixation Distribution Across Regions')
plt.xlabel('Region')
plt.ylabel('Number of Gaze Points')
plt.xticks(rotation=90)
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
plt.show()

# Plot the EAR data
plt.figure(figsize=(10, 6))
plt.plot(timestamps, ears, label='EAR', color='b')
plt.xlabel('Time (seconds)')
plt.ylabel('Eye Aspect Ratio (EAR)')
plt.title(f'EAR over time with {str(blink_count)} blink(s) ')
plt.legend()

# Annotate the blinks on the plot
for blink_time in blink_times:
    plt.axvline(blink_time, color='r', linestyle='--')

# To avoid duplicate labels in the legend, add a custom legend entry
if not blink_times.empty:
    plt.axvline(blink_times[0], color='r', linestyle='--', label='Blink')  # only label once

plt.legend()
plt.show()

# Probability of gaze fixation
total_fixations = len(coordinates)
region_probabilities = {name: (count / total_fixations) * 100 for name, count in region_counts.items()}
table_data = tabulate(region_probabilities.items(), headers=['Region', 'Probability (%)'], tablefmt='fancy_grid')


def show_table_popup():
    root = tk.Tk()
    root.title("Region Probability Table")

    # Create a scrollable text widget
    text_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=45, height=35)
    text_widget.pack(padx=10, pady=10)

    text_widget.insert(tk.END, table_data)
    text_widget.configure(state='disabled')

    root.mainloop()


show_table_popup()

plt.figure(figsize=(10, 6))
probabilities = list(region_probabilities.values())
plt.bar(labels, probabilities, color='skyblue')
plt.title('Gaze Fixation Probability Distribution Across Regions')
plt.xlabel('Region')
plt.ylabel('Probability')
plt.xticks(rotation=90)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', color='black', alpha=0.7)

plt.xticks(rotation=90)
plt.show()
