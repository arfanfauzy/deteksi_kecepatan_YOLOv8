import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker  # Mengimpor modul tracker
import time
from math import dist

# Memuat model YOLO
model = YOLO('yolov8s.pt')

# Membuka file video
cap = cv2.VideoCapture('testing.mp4')

# Membaca nama kelas dari coco.txt
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

# Inisialisasi tracker
tracker = Tracker()

# Menetapkan posisi garis untuk menghitung kendaraan
cy1 = 322
cy2 = 368

# Menetapkan offset untuk mendeteksi kendaraan yang melewati garis
offset = 6

# Kamus untuk menyimpan ID kendaraan dan waktu untuk menghitung kecepatan
vh_down = {}
counter = []

vh_up = {}
counter1 = []

while True:
    ret, frame = cap.read()  # Membaca frame dari video
    if not ret:
        break
    count += 1
    if count % 3 != 0:  # Memproses setiap frame ke-3 untuk mengurangi beban komputasi
        continue
    frame = cv2.resize(frame, (1020, 500))  # Mengubah ukuran frame

    # Melakukan prediksi menggunakan model YOLO
    results = model.predict(frame)
    a = results[0].boxes.data.cpu().numpy()  # Mengonversi tensor ke array numpy setelah dipindahkan ke CPU
    px = pd.DataFrame(a).astype("float")

    list = []

    # Iterasi melalui objek yang terdeteksi
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:  # Hanya mempertimbangkan mobil
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)  # Memperbarui tracker dengan bounding boxes yang terdeteksi
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2  # Menghitung koordinat x tengah dari bounding box
        cy = int(y3 + y4) // 2  # Menghitung koordinat y tengah dari bounding box

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)  # Menggambar bounding box

        # Memeriksa apakah kendaraan melewati garis 1 (arah ke bawah)
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            vh_down[id] = time.time()  # Menyimpan waktu ketika kendaraan melewati garis 1
        if id in vh_down:
            if cy2 < (cy + offset) and cy2 > (cy - offset):  # Memeriksa apakah kendaraan melewati garis 2
                elapsed_time = time.time() - vh_down[id]  # Menghitung waktu yang berlalu
                if counter.count(id) == 0:
                    counter.append(id)  # Menambahkan ID kendaraan ke daftar counter
                    distance = 10  # Jarak antara garis dalam meter
                    a_speed_ms = distance / elapsed_time  # Menghitung kecepatan dalam m/s
                    a_speed_kh = a_speed_ms * 3.6  # Mengonversi kecepatan ke km/h
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)  # Menggambar lingkaran di tengah
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Memeriksa apakah kendaraan melewati garis 2 (arah ke atas)
        if cy2 < (cy + offset) and cy2 > (cy - offset):
            vh_up[id] = time.time()  # Menyimpan waktu ketika kendaraan melewati garis 2
        if id in vh_up:
            if cy1 < (cy + offset) and cy1 > (cy - offset):  # Memeriksa apakah kendaraan melewati garis 1
                elapsed1_time = time.time() - vh_up[id]  # Menghitung waktu yang berlalu
                if counter1.count(id) == 0:
                    counter1.append(id)  # Menambahkan ID kendaraan ke daftar counter
                    distance1 = 10  # Jarak antara garis dalam meter
                    a_speed_ms1 = distance1 / elapsed1_time  # Menghitung kecepatan dalam m/s
                    a_speed_kh1 = a_speed_ms1 * 3.6  # Mengonversi kecepatan ke km/h
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)  # Menggambar lingkaran di tengah
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Menggambar garis pada frame
    cv2.line(frame, (274, cy1), (814, cy1), (255, 255, 255), 1)
    cv2.putText(frame, 'L1', (277, 320), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)
    cv2.putText(frame, 'L2', (182, 367), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Menampilkan jumlah kendaraan yang bergerak ke atas dan ke bawah
    d = len(counter)
    u = len(counter1)
    cv2.putText(frame, 'goingdown:-' + str(d), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, 'goingup:-' + str(u), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Menampilkan frame
    cv2.imshow("Sistem Deteksi Kecepatan", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Keluar jika tombol ESC ditekan
        break

cap.release()  # Melepaskan objek video capture
cv2.destroyAllWindows()  # Menutup semua jendela OpenCV
