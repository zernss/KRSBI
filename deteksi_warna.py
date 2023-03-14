import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    # Baca frame dari kamera
    ret, frame = cap.read()

    # Konversi gambar ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Tentukan rentang warna untuk di-deteksi (misalnya, warna kulit)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Hitung frekuensi warna yang mendominasi
    hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
    dominant_color = np.argmax(hist)

    # Tampilkan gambar
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)

    # Jika tombol 'q' ditekan, keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
