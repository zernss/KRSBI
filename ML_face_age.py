import cv2
import numpy as np
from keras.models import load_model

# Load face and age models
face_model = load_model('face_model.h5')
age_model = load_model('age_model.h5')

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load image
img = cv2.imread('test_image.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Loop through all faces and predict age
for (x, y, w, h) in faces:
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (128, 128))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)
    age = age_model.predict(face)[0][0]
    age = int(age)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, str(age), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()