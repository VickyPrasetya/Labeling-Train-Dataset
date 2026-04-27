from ultralytics import YOLO

def main():

    model = YOLO("yolo11s.pt")

    results = model.train(
        # Masukkan path lengkap menuju file data.yaml kamu
        data="D:/Data Vicky/Dataset Foot Ulcer/Foot Ulcer.v1i.yolov11/data.yaml",       
        epochs=10,              
        imgsz=640,              
        batch=8,               
        device="cpu",              
        project="VISIOPED",     
        name="training_ulang",   
        plots=True              
    )

    print("Proses training selesai!")

if __name__ == '__main__':
    main()