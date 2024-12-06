# Face Attendance System

This project is a **Face Attendance System** built using Python and OpenCV. The system recognizes faces through a webcam, logs attendance based on face recognition, and provides a web interface to view the attendance records. The project includes three main components: face data collection, real-time face recognition, and attendance display via Streamlit.

---

## Project Components

### 1. **Add Faces (add_faces.py)**

This script is responsible for capturing and saving face images for training the recognition model. It detects faces in real-time using OpenCV’s Haar Cascade Classifier and stores the cropped face images into a pickle file, along with corresponding labels (names).

**How it works:**
- The user enters their name.
- The script starts capturing frames from the webcam and detects faces.
- Captured face images are resized and saved.
- Faces are stored in a `faces_data.pkl` file, and corresponding names are saved in `names.pkl`.

**Key functionality:**
- Face detection with OpenCV.
- Capture and save face images with labels.
- Store face data in pickle files.

```python
# Add Faces (add_faces.py)
import cv2
import pickle
import numpy as np
import os

# Initialize webcam and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []
i = 0
name = input("Enter Your Name: ")

# Capture face data
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:
        break

# Save face data
video.release()
cv2.destroyAllWindows()
faces_data = np.asarray(faces_data).reshape(100, -1)

if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names += [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)

# Streamlit Web Interface (app.py)
import streamlit as st
import pandas as pd
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Get current date and time
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

# Auto-refresh interval
count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

if count == 0:
    st.write("Count is zero")
elif count % 3 == 0 and count % 5 == 0:
    st.write("FizzBuzz")
elif count % 3 == 0:
    st.write("Fizz")
elif count % 5 == 0:
    st.write("Buzz")
else:
    st.write(f"Count: {count}")

# Display Attendance
df = pd.read_csv("Attendance/Attendance_" + date + ".csv")
st.dataframe(df.style.highlight_max(axis=0))

# Streamlit Web Interface (app.py)
import streamlit as st
import pandas as pd
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Get current date and time
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

# Auto-refresh interval
count = st_autorefresh(interval=2000, limit=100, key="fizzbuzzcounter")

if count == 0:
    st.write("Count is zero")
elif count % 3 == 0 and count % 5 == 0:
    st.write("FizzBuzz")
elif count % 3 == 0:
    st.write("Fizz")
elif count % 5 == 0:
    st.write("Buzz")
else:
    st.write(f"Count: {count}")

# Display Attendance
df = pd.read_csv("Attendance/Attendance_" + date + ".csv")
st.dataframe(df.style.highlight_max(axis=0))

