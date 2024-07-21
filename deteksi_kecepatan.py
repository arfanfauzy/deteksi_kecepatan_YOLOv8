import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import time

model = YOLO('yolov8s.pt')  # Menginisialisasi model YOLO

cap = cv2.VideoCapture('testing.mp4')  # Membuka video untuk diproses

# Membaca file COCO class names
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
tracker = Tracker()  # Menginisialisasi tracker

# Koordinat garis deteksi
cy1 = 322
cy2 = 368
offset = 6

# Dictionaries dan lists untuk melacak kendaraan yang bergerak ke bawah dan ke atas
vh_down = {}
counter = []

vh_up = {}
counter1 = []

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Keluar dari loop jika video sudah selesai

    count += 1
    if count % 3 != 0:
        continue  # Mengolah setiap frame ke-3

    frame = cv2.resize(frame, (1020, 500))  # Meresize frame video

    results = model.predict(frame)  # Memprediksi objek dalam frame
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list = []

    # Mengolah hasil prediksi
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)  # Mengupdate tracker dengan bounding box terbaru

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)  # Menggambar bounding box

        # Mendeteksi kecepatan kendaraan yang bergerak ke bawah
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            vh_down[id] = time.time()
        
        if id in vh_down:
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                elapsed_time = time.time() - vh_down[id]
                if counter.count(id) == 0:
                    counter.append(id)
                    distance = 10  # meter
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Mendeteksi kecepatan kendaraan yang bergerak ke atas
        if cy2 < (cy + offset) and cy2 > (cy - offset):
            vh_up[id] = time.time()

        if id in vh_up:
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                elapsed1_time = time.time() - vh_up[id]
                if counter1.count(id) == 0:
                    counter1.append(id)
                    distance1 = 10  # meter
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Menampilkan garis deteksi
    cv2.line(frame, (274, cy1), (814, cy1), (255, 255, 255), 1)
    cv2.putText(frame, 'L1', (277, 320), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.line(frame, (177, cy2), (927, cy2), (255, 255, 255), 1)
    cv2.putText(frame, 'L2', (182, 367), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Menampilkan jumlah kendaraan yang bergerak ke bawah dan ke atas
    d = len(counter)
    u = len(counter1)
    cv2.putText(frame, 'going down: ' + str(d), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, 'going up: ' + str(u), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Menampilkan frame
    cv2.imshow("Deteksi Kecepatan Kendaraan", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
