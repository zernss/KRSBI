import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.models import Model

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load dataset
face_df = pd.read_csv('wider_face.csv')
age_df = pd.read_csv('imdb_wiki.csv')

# Preprocess face data
X = []
y = []
for i in range(len(face_df)):
    filename = face_df.iloc[i]['filename']
    x1 = int(face_df.iloc[i]['x'])
    y1 = int(face_df.iloc[i]['y'])
    x2 = int(x1 + face_df.iloc[i]['width'])
    y2 = int(y1 + face_df.iloc[i]['height'])
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        if x1 <= x and x2 >= x+w and y1 <= y and y2 >= y+h:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (128, 128))
            face = face / 255.0
            X.append(face)
            y.append(1)

        # Preprocess age data
            X_age = []
            y_age = []
for i in range(len(age_df)):
    age
    # Skip data if age is not available
    if np.isnan(age_df.iloc[i]['age']):
        continue
        filename = age_df.iloc[i]['full_path']
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            ace = img[y:y+h, x:x+w]
            face = cv2.resize(face, (128, 128))
            face = face / 255.0
            X_age.append(face)
            y_age.append(age_df.iloc[i]['age'])
            X = np.array(X)
            y = np.array(y)
            X_age = np.array(X_age)
            y_age = np.array(y_age)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            X_age_train, X_age_val, y_age_train, y_age_val = train_test_split(X_age, y_age, test_size=0.2, random_state=42)
            input_shape = (128, 128, 3)
            input_tensor = Input(shape=input_shape)
            x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
            x = Conv2D(64, (3, 3), activation='relu')(x)
            x = Conv2D(128, (3, 3), activation='relu')(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            output_tensor = Dense(1, activation='sigmoid')(x)
            face_model = Model(inputs=input_tensor, outputs=output_tensor)
            face_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            face_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
            face_model.save('face_model.h5')
            age_model.save('age_model.h5')