This project is a custom image recognition and object dectection system built in Python 
using TensorFlow, OpenCV and other libraries.
This project suports dataset creation, model training, and real time object classification using
the webcam or external device with the UI support.

The system was designed as an end to end  machine learning pipeline, starting from raw image 
capture to live inference.


Project Objectives
1- Capture image datasets using the camera
2 - Split the data
3- Train a TensorFlow image classification
4- Perform object recognition
5-Draw bouding boxes
6- Provide modular, extensive architecture suitable for experimentation and coursework


Project Structure

src:
Helper => draw_auto_box.py   Bounding logic, but it is refined in the UI

camera.py                    Handle the phone as a Camera(iPhone)
capture.py                   Core detection logic
detector.py -> not in full use, just a test class 
predict_live_ft.py           Live TensorFlow prediction
split_dataset.py             train/val split
train_ft_classifier.py       TensorFlow model training
ui_app.py                    UI_based app
Object_dectection_project.py main entry point

readme                       requirements


Technologies used

Python 3.x
TensorFlow/ Keras
OpenCV
Numpy
Pillow
Tkinter
Pandas
others....
Visual Studio/VS Code

Dataset Creation

Images are captured using a live a camera feed
capture.py

Features

Capture labaled images per object
Store images in class specific dirs
Designed for rapid dataset expansion


Dataset Splitting

split_dataset.py

structure -> dataset-train & val

A TensorFlow image classification model is trained using captured dataset

train_ft_classifier

Key details:
   uses tranfer learning
   automatically saves trained model
   generates a label mapping file


Live Object Recognition

   predict_live_tf.py

Capabilities
   Live frame capture
   TensorFlow interference per frame
   Confidence-based prediction
   Automatic bouding box rendering

Bouding Boxes
This is draw to dynamically object confidence
  draw_auto_box.py

User Interface
A UI based version of the detector is available
 ui_app.py

 Features 
  Start/Stop detection
  Live video display
  Prediction feedback

Running the project
 From terminal capture -> Object_Detection_Project.py + the name of the object to store and 
                                                         classify

 capture intagrated in main
 Split data: pythong split_data_set.py
 Train the model: python train_ft_classification.py
 Live Prediction: python predict_live_ft.py
 Run app to dectect objects: python ui_app.py

Known Challenges
Camera conflicts between laptop and phone webcam
Dataset imbalace affects prediction
Bouding box scaling during live inference
Environment configuration differences in VS

Future Improvement
Use image classification (YOLO/SSD)
Add confidence thersholds and class filtering
improve dataset augmentation
Export trained model for deployment
Add performance metrics

Accademic Context
The project was developed as part of AI/ML course work, focussed on:
Dataset engineering
Model training
Real time inference
Computer vision challenges


