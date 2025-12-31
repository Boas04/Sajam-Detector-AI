# 🔪 Sajam Detector AI (Traditional Weapon Detection)

Sistem deteksi objek *real-time* untuk mengidentifikasi senjata tajam tradisional Indonesia (**Golok, Celurit, Parang**) menggunakan model *deep learning* **YOLOv11**. Aplikasi ini dibangun dengan **Streamlit** sebagai antarmuka interaktif untuk mendukung pemantauan keamanan secara cerdas.

---

## 👨‍💻 Developer (Kelompok)

| No | Nama                              | NIM        |
|----|-----------------------------------|------------|
| 1  | Abner Boas P. P. Gultom           | 4523210002 |
| 2  | Andika Haikal Syahputra           | 4523210016 |
| 3  | Fajar Istiqomah                   | 4523210045 |
| 4  | Mesak Mychart E. Purba            | 4523210062 |
| 5  | Khalissa Raihanah Azhari          | 4523210122 |


**Institusi:** Universitas Pancasila  

---

## 🧠 Deskripsi
**Sajam Detector AI** merupakan aplikasi berbasis *computer vision* yang dirancang untuk membantu meningkatkan keamanan di ruang publik melalui deteksi otomatis senjata tajam menggunakan kamera pengawas. Sistem ini mampu melakukan deteksi secara *real-time* dan menampilkan hasil berupa *bounding box*, label kelas, serta tingkat kepercayaan (*confidence score*).

---

## ⚙️ Technologies Used
- Python 3.8+  
- Streamlit  
- Ultralytics YOLO (YOLOv11 & YOLOv9)  
- OpenCV  
- Pillow  

---

## 🚀 Getting Started

### Prasyarat
- Python 3.8 atau lebih baru  
- Webcam aktif  

### Instalasi
1. Clone repository atau download folder project.  
2. Instal dependensi:
   ```bash
   pip install streamlit ultralytics opencv-python pillow
3. Jalankan Applikasi
   streamlit run Main.py


## 🧩 Project Structure
- Main.py — Script utama aplikasi (Streamlit)  
- best.pt — Model YOLO hasil training  
- run.bat — Shortcut menjalankan aplikasi  
- README.md


## 🧠 How It Works

- Model YOLOv11 memuat bobot hasil pelatihan (`best.pt`).
- Kamera menangkap frame video secara langsung.
- Setiap frame diproses oleh model untuk mendeteksi objek senjata tajam.
- Jika terdeteksi **Golok**, **Celurit**, atau **Parang**, sistem akan menampilkan *bounding box* dan skor kepercayaan secara visual.

---

## ✨ Features

- Deteksi senjata tajam secara real-time
- Multi-class detection (Golok, Celurit, Parang)
- Tampilan confidence score
- Antarmuka modern berbasis Streamlit

---

## 🎮 Usage

1. Jalankan aplikasi.
2. Izinkan akses kamera.
3. Arahkan objek ke kamera.
4. Sistem akan otomatis mendeteksi dan menampilkan hasil prediksi.

---

## 📝 License

Proyek ini dibuat untuk memenuhi **Tugas Akhir Mata Kuliah Pembelajaran Mesin**  
Program Studi Teknik Informatika – Universitas Pancasila  
Tahun Akademik 2025/2026
