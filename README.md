# CNN_Based_Gaze_Tracking_Interface
This repository contains scripts for collecting eye-tracking data, training a CNN model, and analyzing real-time tracking.

Steps to Get Started

1. **Clone the Repository**:
Clone the repository to your local machine using:
git clone https://github.com/Visal-Vidhya/CNN_Based_Gaze_Tracking_Interface.git

3. **Data Collection**:
Run the Data_collection.py script.
A screen will appear with a grid of 16 cells, with a red pulsating dot for 5 sec, moving sequentially across the cells.
Follow the red dot. The system will capture 256x256 pixel grayscale images of your right eye for each cell as the dot moves.
To collect additional data, simply change the file name for each cycle to avoid overwriting previous images.

5. **Model Training**:
After completing data collection, run Train_&_Analysis.py to train and save the Convolutional Neural Network (CNN) model.
Adjust the hyperparameters in the script to suit your requirements.
Once the training is complete, update the model file name in all relevant tracking and analysis scripts to match the newly trained model.

7. **Real-Time Tracking and Analysis**:
Run Interface.py, this will open the gaze tracking interface to start tracking and analyzing eye movement using the trained model.

8. Accuracy button on the interface will calculate accuracy of the model using a novel trajectory based method, here a blue ball will move accross edge the screen and user has to follow the ball using there gaze until the process is completed.
   
9. **Citation**:
If you find this repository helpful in your research or project, please consider citing it by referencing the repository title, contributors, and the repository URL.
 
**Additional Notes**
Be sure to adjust the file paths and model names as needed throughout the process.
Feel free to experiment with different parameters during model training for better results.

Cheers!
