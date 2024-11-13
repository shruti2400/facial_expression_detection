# Facial Expression Detection

Overview
This project leverages Convolutional Neural Networks (CNNs) to detect emotions from facial expressions in real-time using a webcam. The model was trained on the Kaggle FER-2013 dataset, which consists of grayscale images of faces classified into seven emotion categories:

Angry
Disgust
Fear
Happy
Sad
Surprise
Neutral

dataset: [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)

Model:

The model was built using TensorFlow/Keras.

It uses a Convolutional Neural Network (CNN) architecture.

The model achieved an accuracy of 67% on the test dataset.

The trained model is saved in a .h5 file for future use.

Python Packages:

TensorFlow (for building and training the CNN model)

Keras (for model utilities and preprocessing)

OpenCV (for real-time webcam functionality)

NumPy (for numerical computations)

Pandas (for handling dataframes)

Matplotlib (for plotting graphs, if used)

Keras-Preprocessing (for image data augmentation)

Kaggle (for downloading datasets directly from Kaggle)


Files in This Repository

final_emotion_detection_model67.h5: Trained CNN model with 67% accuracy.

final_facial_expression_model_CNN.py: Python script used for training the model on the FER-2013 dataset.

webcam.py: Script for real-time emotion detection using a webcam.

README.md: Project documentation.
