import tkinter as tk
from tkinter import PhotoImage, messagebox
import subprocess


def run_script(script_name):
    try:
        print(f"Running: {script_name}")
        subprocess.run(["python", script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Function wrappers
def run_gaze_tracking():
    run_script("Gaze_Track_Store.py")


def run_analysis():
    run_script("Heat_map_Analysis.py")


def run_accuracy():
    # Display a message box with instructions
    response = messagebox.askokcancel(
        "Accuracy Test Instructions",
        "Follow the blue ball on the screen until the process is finished.\nWhen you are ready, click OK."
    )
    if response:
        # Only run the script if OK is clicked
        run_script("Trajectory_metrics.py")


def run_videogaze():
    run_script("Video_GazeTrack.py")


def run_videoAnalysis():
    run_script("Video_HeatMap_Analysis.py")


# Create the main application window
root = tk.Tk()
root.title("Gaze Tracking Application")
root.geometry("800x600")
root.configure(bg='#000000')  # Black Background

# Load and resize icons
icon_gaze = PhotoImage(file="icons/gaze.png").subsample(5, 5)
icon_analysis = PhotoImage(file="icons/analysis.png").subsample(5, 5)
icon_accuracy = PhotoImage(file="icons/accuracy.png").subsample(5, 5)
icon_videogaze = PhotoImage(file="icons/videogaze.png").subsample(5, 5)
icon_video_analysis = PhotoImage(file="icons/video_analysis.png").subsample(5, 5)


# Function to create buttons with walnut and pastel green accents
def create_pastel_button(parent, icon, text, command, row, col):
    def on_enter(event):
        button_frame.config(relief="raised", bg="#c8e6c9")  # Light Pastel Green on hover
        icon_button.config(bg="#c8e6c9")
        text_label.config(bg="#c8e6c9", fg="#ffffff")  # White text on hover

    def on_leave(event):
        button_frame.config(relief="flat", bg="#a5d6a7")  # Pastel Green
        icon_button.config(bg="#a5d6a7")
        text_label.config(bg="#a5d6a7", fg="#ffffff")  # White text

    # Frame for 3D effect
    button_frame = tk.Frame(parent, bg='#a5d6a7', relief="flat", bd=2, padx=10, pady=10)

    # Icon Button
    icon_button = tk.Button(button_frame, image=icon, command=command, bg='#a5d6a7', bd=0, relief="flat")
    icon_button.pack(pady=(5, 0))

    # Text Label
    text_label = tk.Label(button_frame, text=text, fg='#ffffff', bg='#a5d6a7', font=('Arial', 12))
    text_label.pack(pady=(0, 5))

    # Grid placement
    button_frame.grid(row=row, column=col, padx=15, pady=15, sticky="nsew")

    # Bind hover effects
    button_frame.bind("<Enter>", on_enter)
    button_frame.bind("<Leave>", on_leave)
    icon_button.bind("<Enter>", on_enter)
    icon_button.bind("<Leave>", on_leave)
    text_label.bind("<Enter>", on_enter)
    text_label.bind("<Leave>", on_leave)


# Create and pack label
Gaze_label = tk.Label(root, text='Gaze Tracking', fg='#ffffff', bg='#000000', font=('Arial black', 30))
Gaze_label.pack(pady=20)

# Create a frame to hold the buttons in a 2x3 grid
button_frame = tk.Frame(root, bg='#000000')
button_frame.pack(expand=True)

# Configure grid rows and columns to be expandable
button_frame.grid_rowconfigure(0, weight=1)
button_frame.grid_rowconfigure(1, weight=1)
button_frame.grid_columnconfigure(0, weight=1)
button_frame.grid_columnconfigure(1, weight=1)
button_frame.grid_columnconfigure(2, weight=1)

# Create buttons in a 2x3 grid
create_pastel_button(button_frame, icon_gaze, "Gaze Tracking", run_gaze_tracking, row=0, col=0)
create_pastel_button(button_frame, icon_analysis, "Analysis", run_analysis, row=0, col=1)
create_pastel_button(button_frame, icon_accuracy, "Accuracy Test", run_accuracy, row=0, col=2)
create_pastel_button(button_frame, icon_videogaze, "Video Gaze Tracking", run_videogaze, row=1, col=0)
create_pastel_button(button_frame, icon_video_analysis, "Video Gaze Analysis", run_videoAnalysis, row=1, col=1)

# Start the GUI event loop
root.mainloop()
