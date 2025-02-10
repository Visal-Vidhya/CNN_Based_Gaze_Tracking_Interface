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

Eye Gaze Tracking using CNN – Open Source Code

Overview

This repository provides the open-source implementation of Eye Gaze Tracking using CNN,

developed as part of our research work:

� Reference Paper:

"Real-Time Gaze Estimation Using Webcam-Based CNN Models for Human-Computer

Interaction"

� Authors: Visal Vidhya, Diego Resende Faria

� Publication Date: February 2025

� Journal: Computers (MDPI)

How to Cite

If you use this code in your research, publications, or projects, you must cite our work as follows:

bibtex

CopyEdit

@article{VidhyaFaria2025,

author = {Visal Vidhya and Diego Resende Faria},
 
title = {Real-Time Gaze Estimation Using Webcam-Based CNN Models for Human-

Computer Interaction},
 
journal = {Computers},

publisher = {MDPI},

year = {2025},

month = {February}

}

� Proper citation is mandatory when using this code.
 
License & Usage Policy

This code is provided strictly for research and personal use only under the following conditions:

✔ Allowed Usage:

✅ Academic research

✅ Personal projects

✅ Non-commercial studies

❌ Prohibited Usage:

⛔ Commercial applications, services, or products

⛔ Industrial deployment or commercial research without explicit permission

⛔ Redistribution of the code without proper citation
 
2
 
Data Privacy & Model Limitations

⚠ Important Note:

• This model is trained on data from only 8 individuals, which may not generalize well to

other users.

• If you wish to use this for broader applications, you might need to retrain the model on a

larger, more diverse dataset.
 
Disclaimer & Liability

⚠ No Warranty & No Guarantee of Accuracy

• This software is provided "as-is" without any guarantees or warranties of accuracy,

performance, or reliability.

• The authors are not responsible for any issues, inaccuracies, or unintended

consequences that may arise from using this code.
 
• Users assume full responsibility for validating and verifying the output before any real-

world application.
 
� By using this code, you agree that the authors are not liable for any direct, indirect,

incidental, or consequential damages resulting from its use.
 
Contributing

Contributions are welcome for academic and research improvements only.

To contribute.
 
Contact & Support

For any academic inquiries or collaboration requests, please contact:

✉ [ vishalvidhya95@gmail.com and/or fariadiego@gmail.com ]

� Remember: This code is NOT for commercial use!
 

Cheers!
