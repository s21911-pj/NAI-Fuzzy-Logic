# Authors:
# Karol Kraus s20687
# Piotr Mastalerz s21911

# application recognize hand gestures thanks to that we can control Spotify player
# list of gestures and actions:

# thumbs up - volume UP
# thumbs down - volume DOWN
# peace - next song
# stop - stop song

# environmental instructions
# create venv
#   python3 -m venv venv
# activate venv
#   source venv/bin/activate
# install packages
#   pip3 install -r requirements.txt
# run app
#   python3 main.py


# import necessary packages for hand gesture recognition project using Python OpenCV
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import pyautogui as keyboard

# initialize mediapipe
# MediaPipe is a customizable machine learning solutions framework developed by Google.
# It is an open-source and cross-platform framework, and it is very lightweight.
# MediaPipe comes with some pre-trained ML solutions such as face detection, pose estimation,
# hand recognition, object detection, etc.

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize Tensorflow
# Load the gesture recognizer model (we load the TensorFlow pre-trained model)
# The model can recognize 10 different gestures.

model = load_model('mp_hand_gesture')

# Load class names
# Gesture.names file contains the name of the gesture classes
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()

# We create a VideoCapture object and pass an argument ‘0’ - It is the camera ID of the system.
# In this case, we have 1 webcam connected with the system.
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)


    className = ''
    classID = None

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:

                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])

            classID = np.argmax(prediction)
            className = classNames[classID]

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)

    # according to the gesture we control the spotify player

    if classID == 2:
        keyboard.hotkey("command", "up")
    elif classID == 3:
        keyboard.hotkey("command", "down")
    elif classID == 5:
        keyboard.press("space")
    elif classID == 1:
        keyboard.hotkey("command", "right")
    else:
        pass


    cv2.imshow("Gesture Handler", frame)

    # function keeps the window open until the key ‘q’ is pressed.

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
