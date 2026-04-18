import cv2

print("Mencari kamera yang aktif di laptop...")

# Ngecek dari index 0 sampai 4
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        success, frame = cap.read()
        if success:
            print(f"✅ Kamera ketemu di Index: {i}")
            # Tampilin window-nya
            cv2.imshow(f"Kamera Index {i} - Tekan tombol apa aja buat lanjut", frame)
            cv2.waitKey(0) # Program berhenti nunggu lu pencet tombol di keyboard
            cv2.destroyWindow(f"Kamera Index {i} - Tekan tombol apa aja buat lanjut")
        cap.release()

print("Selesai ngecek kamera!")