from ultralytics import YOLO
import cv2

model = YOLO("C:/Users/User/OneDrive/Documents/Foot Ulcer/runs/detect/hasil_training/percobaan_pertama4/weights/best.pt")

CAMERA_INDEX = 1

def buka_kamera(index):
    backend_candidates = [
        ("default", None),
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
    ]

    for backend_name, backend in backend_candidates:
        if backend is None:
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index, backend)

        if not cap.isOpened():
            cap.release()
            continue

        success, _ = cap.read()
        if success:
            print(f"Kamera terbuka lewat backend {backend_name} di index {index}.")
            return cap

        cap.release()

    return None

print("Isi model.names:", model.names)

cap = buka_kamera(CAMERA_INDEX)

if cap is None:
    raise SystemExit("Kamera tidak bisa dibuka.")

jumlah_0 = 0
jumlah_1 = 0
jumlah_2 = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame, conf=0.1, show=False)

    terdeteksi_0 = False
    terdeteksi_1 = False
    terdeteksi_2 = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = str(model.names[cls])

            print(f"cls={cls}, class_name={class_name}, conf={conf:.2f}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if cls == 0:
                terdeteksi_0 = True
            elif cls == 1:
                terdeteksi_1 = True
            elif cls == 2:
                terdeteksi_2 = True

    if terdeteksi_0:
        jumlah_0 += 1
    if terdeteksi_1:
        jumlah_1 += 1
    if terdeteksi_2:
        jumlah_2 += 1

    cv2.putText(frame, f"0: {jumlah_0}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"1: {jumlah_1}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"2: {jumlah_2}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    hasil_akhir = None
    if jumlah_0 >= 5:
        hasil_akhir = "tidak diabetes"
    elif jumlah_1 >= 5:
        hasil_akhir = "hanya luka biasa"
    elif jumlah_2 >= 5:
        hasil_akhir = "terindikasi diabetes"

    if hasil_akhir:
        cv2.putText(frame, hasil_akhir, (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("DFU Scanner - Realtime USB", frame)
        print(hasil_akhir)
        cv2.waitKey(8000)
        break

    cv2.imshow("DFU Scanner - Realtime USB", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()