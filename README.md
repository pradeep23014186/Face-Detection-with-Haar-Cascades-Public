# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows

## Program:

### I) Load and Display Images
```py

import cv2
import matplotlib.pyplot as plt
import numpy as np

img1_gray = cv2.imread("image_01.png", cv2.IMREAD_GRAYSCALE)
img2_gray = cv2.imread("image_02.png", cv2.IMREAD_GRAYSCALE)
img3_gray = cv2.imread("image_03.png", cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(18,6))

plt.subplot(131); plt.title("Image 1"); plt.imshow(img1_gray, cmap = "gray")
plt.subplot(132); plt.title("Image 2"); plt.imshow(img2_gray, cmap = "gray")
plt.subplot(133); plt.title("Image 3"); plt.imshow(img3_gray, cmap = "gray")

plt.figure(figsize=(18,6))

plt.subplot(131); plt.title("Image 1"); plt.imshow(cv2.resize(img1_gray, (1000, 1000)), cmap = "gray")
plt.subplot(132); plt.title("Image 2"); plt.imshow(cv2.resize(img2_gray, (1000, 1000)), cmap = "gray")
plt.subplot(133); plt.title("Image 3"); plt.imshow(cv2.resize(img3_gray, (1000, 1000)), cmap = "gray")

```

### II) Load Haar Cascade Classifiers
```py

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

```

### III) Perform Face Detection in Images
```py

def detect_face(img):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, 1.3, 5)

    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x+w, y+h), (255,255,255), 10)
    return face_img

result1 = detect_face(img2_gray)
plt.imshow(result1, cmap = 'gray')
plt.title("IMAGE 1")

result2 = detect_face(img1_gray)
plt.imshow(result2, cmap = 'gray')
plt.title("IMAGE 2")

result3 = detect_face(img3_gray)
plt.imshow(result3, cmap = 'gray')
plt.title("IMAGE 3")

```

### IV) Perform Eye Detection in Images
```py

eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

def detect_eye(img):
    eye_img = img.copy()
    eye_circle = eye_cascade.detectMultiScale(eye_img, 1.3, 8)
    for (x,y,w,h) in eye_circle:
        h = round(h/2)
        w = round(w/2)
        cv2.circle(eye_img, (x+w, y+h), h, (255,255,255), 5)
    return eye_img

result1 = detect_eye(img2_gray)
plt.imshow(result1, cmap = "gray")
plt.title("IMAGE 1")

result2 = detect_eye(img1_gray)
plt.imshow(result2, cmap = "gray")
plt.title("IMAGE 2")

```

### V) Perform Face Detection on Real-Time Webcam Video
```py

cap = cv2.VideoCapture(0)

plt.ion()
flag, ax = plt.subplots()

ret, frame = cap.read(0)
face = detect_face(frame)
im = ax.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
plt.title("Video Face Detection")

while True:
    ret, frame = cap.read(0)
    face = detect_face(face)
    im.set_data(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    plt.pause(0.10)

cap.release()
plt.close()

```
