import cv2

def main():
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    if not cap.isOpened():
        print("Error: Gagal membuka kamera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Gagal membaca frame dari kamera.")
            break

        mirrored_frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(mirrored_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.putText(
            mirrored_frame,
            f'Face Detected: {len(faces)}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow('Face Detection Count', mirrored_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()