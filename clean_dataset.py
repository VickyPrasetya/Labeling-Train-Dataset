import os

# Path utama dataset kamu (sesuai gambar)
dataset_path = r"D:\Data Vicky\Dataset Foot Ulcer\Foot Ulcer.v1i.yolov11"

# Folder pembagian data YOLO
splits = ['train', 'valid', 'test']

# Ekstensi gambar yang umum digunakan
image_exts = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']

deleted_count = 0
updated_count = 0

print("Memulai proses pembersihan dataset...")

for split in splits:
    labels_dir = os.path.join(dataset_path, split, 'labels')
    images_dir = os.path.join(dataset_path, split, 'images')

    # Lewati jika foldernya tidak ada
    if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
        continue

    for txt_file in os.listdir(labels_dir):
        # Abaikan file selain .txt dan abaikan file classes.txt jika ada
        if not txt_file.endswith('.txt') or txt_file == 'classes.txt':
            continue

        txt_path = os.path.join(labels_dir, txt_file)
        base_name = os.path.splitext(txt_file)[0]

        # Baca isi file label
        with open(txt_path, 'r') as file:
            lines = file.readlines()

        has_label_1 = False
        new_lines = []
        is_modified = False

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            class_id = int(parts[0])

            # Cek apakah ada label 1
            if class_id == 1:
                has_label_1 = True
                break  # Langsung berhenti, seluruh file ini dan gambarnya akan dihapus
            elif class_id == 2:
                parts[0] = "1"  # Ubah class id 2 menjadi 1
                is_modified = True

            # Gabungkan kembali barisnya
            new_lines.append(" ".join(parts) + "\n")

        # Eksekusi berdasarkan kondisi
        if has_label_1:
            # 1. Hapus file .txt (label)
            os.remove(txt_path)

            # 2. Cari dan hapus file gambar pasangannya di folder images
            for ext in image_exts:
                img_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(img_path):
                    os.remove(img_path)
                    break
            
            deleted_count += 1
        else:
            # Jika tidak dihapus dan ada label 2 yang diubah menjadi 1, timpa file dengan data baru
            if is_modified:
                with open(txt_path, 'w') as file:
                    file.writelines(new_lines)
            updated_count += 1

print("\n--- PROSES SELESAI ---")
print(f"Total gambar & label yang DIHAPUS (karena mengandung label 1): {deleted_count}")
print(f"Total label yang DIPERBARUI (label 2 diubah jadi 1): {updated_count}")