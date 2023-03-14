import cv2
import numpy as np

# Fungsi untuk menghitung jari-jari yang diangkat
def count_fingers(contour, threshold):
    # Buat poligon dari kontur
    hull = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull)
    if len(hull) > 3 and cv2.isContourConvex(largest_contour):
        defects = cv2.convexityDefects(contour, hull)

    # Inisialisasi variabel jumlah jari dan posisi pusat tangan
    finger_count = 0
    center = None

    # Loop melalui cacat untuk menemukan jari-jari yang diangkat
    if defects is not None:
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            # Hitung panjang sisi segitiga
            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

            # Hitung sudut antara dua sisi segitiga
            angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c)) * 180/np.pi

            # Jika sudut lebih kecil dari ambang batas, maka itu adalah jari
            if angle <= threshold:
                finger_count += 1

                # Jika ini adalah jari pertama, simpan posisinya sebagai pusat tangan
                if finger_count == 1:
                    center = far

    return finger_count, center

cap = cv2.VideoCapture(0)

while(True):
    # Baca frame dari kamera
    ret, frame = cap.read()

    # Ubah ke skala abu-abu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduksi noise menggunakan filter Gaussian
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding adaptif untuk meningkatkan kontras
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Dapatkan kontur dari gambar hasil thresholding
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop melalui semua kontur untuk menemukan yang paling besar (tangan)
    max_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    # Jika kontur tangan ditemukan, hitung jumlah jari
    if largest_contour is not None:
        finger_count, center = count_fingers(largest_contour, 90)

        # Gambar lingkaran pada pusat tangan
        if center is not None:
            cv2.circle(frame, center, 8, (0, 0, 255), thickness=3)

    # Tampilkan jumlah jari yang diangkat pada layar
    cv2.putText(frame, str(finger_count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Tampilkan gambar pada layar
    cv2.imshow('Hand Gesture Recognition', frame)

# Tunggu tombol keyboard untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()