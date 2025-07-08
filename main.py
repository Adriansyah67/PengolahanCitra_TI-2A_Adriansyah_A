import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# 1. Muat Gambar dan Konversi ke Encoding
path = 'images_attendance'
images = []
classNames = []
myList = os.listdir(path)
print("Memuat gambar untuk encoding...")
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(f"Daftar Nama: {classNames}")

def findEncodings(images):
    """Fungsi untuk menghasilkan encoding dari daftar gambar."""
    encodeList = []
    for img in images:
        # Konversi gambar dari BGR ke RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Temukan encoding wajah
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError as e:
            print(f"Error: Tidak ada wajah yang terdeteksi pada salah satu gambar. {e}")
    return encodeList

def markAttendance(name):
    """Fungsi untuk mencatat kehadiran ke dalam file CSV."""
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        
        # Cek jika nama belum tercatat hari ini
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            print(f"Kehadiran {name} telah dicatat pada {dtString}")

print("Memulai proses encoding...")
encodeListKnown = findEncodings(images)
print('Proses Encoding Selesai.')

# 2. Inisialisasi Webcam
cap = cv2.VideoCapture(0)
print("Webcam diaktifkan. Mulai deteksi wajah...")

while True:
    success, img = cap.read()
    if not success:
        print("Gagal membaca frame dari webcam.")
        break
    
    # Perkecil ukuran gambar untuk mempercepat proses
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Temukan lokasi wajah dan encoding di frame saat ini
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # 3. Bandingkan Wajah dan Tandai
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            
            # Gambar kotak di sekitar wajah
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4 # Kembalikan ke ukuran asli
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            # Catat kehadiran
            markAttendance(name)

    # Tampilkan hasil
    cv2.imshow('Webcam', img)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan webcam dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
print("Program selesai.")