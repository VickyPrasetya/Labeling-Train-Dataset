from ultralytics import YOLO

def main():

    model = YOLO("yolo11s.pt")

    results = model.train(
        data="data.yaml",       
        epochs=50,              
        imgsz=640,              
        batch=8,               
        device="cpu",              
        project="VISIOPED",     
        name="training_awal",   
        plots=True              
    )

    print("Proses training selesai!")

if __name__ == '__main__':
    main()