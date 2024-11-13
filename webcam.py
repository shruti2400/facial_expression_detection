import cv2
import numpy as np
from keras.models import load_model
import time

# Load the trained model
model = load_model('final_emotion_detection_model67.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the webcam using DirectShow (which is often more reliable on Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit the webcam.")

# Set a timeout to avoid infinite loop
timeout = time.time() + 10  # 10 seconds timeout

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        
        # Break after 10 seconds if no frame is captured
        if time.time() > timeout:
            print("Exiting due to timeout...")
            break

    # Convert to grayscale for emotion detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Check if the frame is correctly captured
    if gray_frame is None or gray_frame.size == 0:
        print("Error: Empty frame captured")
        continue

    # Resize the frame for the model
    resized_frame = cv2.resize(gray_frame, (48, 48)).reshape(1, 48, 48, 1) / 255.0

    # Predict emotion using the model
    predictions = model.predict(resized_frame)
    max_index = np.argmax(predictions[0])
    emotion = emotion_labels[max_index]

    # Display the emotion label on the screen
    cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the webcam feed
    cv2.imshow('Emotion Detection', frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting on user request...")
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()