from ultralytics import YOLO

def main():
    print("⏳ Mendownload model dasar YOLOv11s...")
    # Kita menggunakan YOLOv11s versi 'small' (yolov11s.pt) agar tidak terlalu berat
    model = YOLO("yolo11s.pt")

    print("🚀 Memulai proses training...")
    
    # ==========================================
    # PENGATURAN TRAINING
    # ==========================================
    # Pastikan path data.yaml di bawah ini sesuai dengan lokasi komputermu
    results = model.train(
        data="C:/Users/User/OneDrive/Documents/Foot Ulcer - Copy/Foot Ulcer.v2i.yolov11/data.yaml", # GANTI INI dengan path ke data.yaml kamu
        epochs= 100,     # Jumlah putaran belajar. Coba 50 dulu.
        imgsz=640,       # Ukuran gambar (standar Roboflow biasanya 640)
        batch=8,         # Jumlah gambar yang diproses sekaligus. 
                         # (Jika komputermu nge-lag/error kehabisan memori, turunkan jadi 4 atau 2)
        device="cpu",    # HAPUS BARIS INI JIKA KAMU PUNYA GPU NVIDIA (RTX/GTX) AGAR LEBIH CEPAT
        project="hasil_training", # Nama folder tempat model disimpan
        name="percobaan_pertama"  # Nama sub-folder
    )
    
    print("✅ Training selesai!")

if __name__ == '__main__':
    # Blok ini wajib ada di Windows agar proses training tidak error
    main()