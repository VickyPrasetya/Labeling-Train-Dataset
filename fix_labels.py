import os

# Tentukan lokasi folder label kamu (Sesuaikan jika ada folder train/val)
# Jika ada folder 'val', kamu perlu menjalankan script ini dua kali (ubah path-nya)
label_folder = r"D:\Data Vicky\Dataset Foot Ulcer\Dataset_YOLO\labels\val"

# Hitung jumlah file yang diubah
count = 0

for filename in os.listdir(label_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(label_folder, filename)
        
        with open(filepath, "r") as file:
            lines = file.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                class_id = int(parts[0])
                
                # --- LOGIKA PENGUBAHAN ---
                # Ganti 14 jadi 0, dan 15 jadi 1
                if class_id == 15:
                    parts[0] = "0"
                elif class_id == 16:
                    parts[0] = "1"
                
                new_line = " ".join(parts) + "\n"
                new_lines.append(new_line)
        
        # Tulis ulang file dengan label yang sudah diperbaiki
        with open(filepath, "w") as file:
            file.writelines(new_lines)
            count += 1

print(f"Selesai! Berhasil memperbarui {count} file label.")