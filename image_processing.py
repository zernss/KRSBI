import cv2
import dlib
import numpy as np

# Load detector dan predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load classifier
classifier = cv2.CascadeClassifier("haarcascade_eye.xml")

# Buka kamera
cap = cv2.VideoCapture(0)

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()

    # Ubah ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah menggunakan detector
    faces = detector(gray)

    # Loop melalui setiap wajah yang terdeteksi
    for face in faces:
        # Dapatkan koordinat landmark pada wajah menggunakan predictor
        landmarks = predictor(gray, face)

        # Ambil titik landmark pada mata dan hidung
        left_eye =  []
        right_eye =[] 
        nose_bridge =[] 

        for n in range(36, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            if n < 42:
                left_eye.append((x, y))
            elif n < 48:
                right_eye.append((x, y))
            elif n == 27:
                nose_bridge.append((x, y))

        # Dapatkan koordinat bounding box pada mata menggunakan classifier
        left_eye_box = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        right_eye_box = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Periksa apakah bounding box mata ditemukan
        if len(left_eye_box) > 0 and len(right_eye_box) > 0:
            cv2.putText(frame, "Using Glasses", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Not Using Glasses", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Gambar garis pada mata dan hidung
        cv2.polylines(frame, [np.array(left_eye, dtype=np.int32)], True, (0, 255, 255), 1)
        cv2.polylines(frame, [np.array(right_eye, dtype=np.int32)], True, (0, 255, 255), 1)
        cv2.polylines(frame, [np.array(nose_bridge, dtype=np.int32)], True, (0, 255, 255), 1)

    # Tampilkan frame
    cv2.imshow("Frame", frame)

    # Jika tombol 'q' ditekan, keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Hentikan kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
