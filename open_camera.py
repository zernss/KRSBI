import cv2 #mengimport library opencv
cap = cv2.VideoCapture(0)
while True: #membuat looping agar kamera tetap menyala selama tidak diberhentikan
    x,frame = cap.read()
    if x:
        cv2.imshow("TESTING", frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):#jika menekan tombol x pada keyboard camera akan terhenti
        break
cap.release()
cv2.destroyAllWindows()#menutup windows yang terbuka saat kamera hidup

