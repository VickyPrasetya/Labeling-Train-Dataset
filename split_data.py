import os
import shutil
import random

folder_images = "Images"
folder_labels = "Labels"

folder_yolo = "Dataset_YOLO"

folders = ['images/train', 'images/val', 'labels/train', 'labels/val']
for folder in folders:
    os.makedirs(os.path.join(folder_yolo, folder), exist_ok=True)

semua_gambar = [f for f in os.listdir(folder_images) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(semua_gambar)

batas = int(len(semua_gambar) * 0.8)
train_imgs = semua_gambar[:batas]
val_imgs = semua_gambar[batas:]

def pindah_file(daftar_gambar, tipe):
    for img in daftar_gambar:
        shutil.copy(os.path.join(folder_images, img), os.path.join(folder_yolo, 'images', tipe, img))

        file_txt = os.path.splitext(img)[0] + ".txt"
        path_txt = os.path.join(folder_labels, file_txt)
        
        if os.path.exists(path_txt):
            shutil.copy(path_txt, os.path.join(folder_yolo, 'labels', tipe, file_txt))
        else:
            print(f"Peringatan: File label untuk {img} tidak ditemukan di folder Labels!")

print("Sedang memproses pemisahan dataset (80% Train, 20% Val)...")
pindah_file(train_imgs, 'train')
pindah_file(val_imgs, 'val')
print("Selesai! Folder 'Dataset_YOLO' berhasil dibuat.")