import dlib
import cv2
import numpy as np

# load face detector dan face landmark predictor dari dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# load gambar dan konversi ke grayscale
img = cv2.imread("C:\College Things\KRSBI\image2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# deteksi wajah pada gambar menggunakan face detector dari dlib
faces = detector(gray)

# loop over setiap wajah yang terdeteksi
for face in faces:
    # deteksi titik-titik landmark pada wajah menggunakan face landmark predictor dari dlib
    landmarks = predictor(gray, face)

    # ambil koordinat-koordinat titik yang membentuk mata
    left_eye_coords = []
    right_eye_coords = []
    for i in range(36, 42):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        left_eye_coords.append((x, y))
    for i in range(42, 48):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        right_eye_coords.append((x, y))

    # buat poligon pada gambar yang menandakan posisi mata
    cv2.polylines(img, [np.array(left_eye_coords, dtype=np.int32)], True, (0, 255, 255), 1)
    cv2.polylines(img, [np.array(right_eye_coords, dtype=np.int32)], True, (0, 255, 255), 1)

# tampilkan gambar hasil deteksi
cv2.imshow("Detected Eyes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()