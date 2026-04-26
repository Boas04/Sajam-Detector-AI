# Panduan Kontribusi

Terima kasih telah tertarik untuk berkontribusi pada **Sajam Detector AI**! 🎉  
Panduan berikut membantu Anda memahami cara terbaik untuk berpartisipasi.

---

## 🗂️ Cara Berkontribusi

### 1. Laporkan Bug
Jika menemukan bug, buat [Issue baru](https://github.com/Boas04/Sajam-Detector-AI/issues/new/choose) menggunakan template **Bug Report** dan sertakan:
- Deskripsi singkat masalah
- Langkah-langkah untuk mereproduksi
- Perilaku yang diharapkan vs yang terjadi
- Screenshot (jika relevan)
- Versi OS, Python, dan library

### 2. Usulkan Fitur Baru
Punya ide pengembangan? Buka [Issue baru](https://github.com/Boas04/Sajam-Detector-AI/issues/new/choose) dengan template **Feature Request** dan jelaskan:
- Masalah yang ingin diselesaikan
- Solusi yang diusulkan
- Konteks atau referensi tambahan

### 3. Kirim Pull Request
1. **Fork** repository ini
2. Buat branch baru dari `main`:
   ```bash
   git checkout -b feat/nama-fitur-anda
   ```
3. Lakukan perubahan dan **commit** dengan pesan yang jelas:
   ```bash
   git commit -m "feat: tambahkan deteksi kelas baru"
   ```
4. **Push** ke fork Anda:
   ```bash
   git push origin feat/nama-fitur-anda
   ```
5. Buka **Pull Request** ke branch `main` menggunakan template yang tersedia

---

## ✅ Standar Kode

- Gunakan **Python 3.8+**
- Ikuti gaya penulisan yang sudah ada di `Main.py`
- Tambahkan komentar untuk bagian yang kompleks
- Pastikan aplikasi tetap berjalan dengan `streamlit run Main.py` tanpa error

---

## 🔍 Lingkungan Pengembangan

```bash
# Clone repo
git clone https://github.com/Boas04/Sajam-Detector-AI.git
cd Sajam-Detector-AI

# Buat virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instal dependensi
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run Main.py
```

---

## 📋 Pesan Commit

Gunakan format berikut untuk pesan commit:

| Prefix | Penggunaan |
|--------|------------|
| `feat:` | Fitur baru |
| `fix:` | Perbaikan bug |
| `docs:` | Perubahan dokumentasi |
| `refactor:` | Refactoring kode |
| `chore:` | Tugas maintenance |

---

## 🤝 Kode Etik

Dengan berkontribusi, Anda setuju untuk mengikuti [Kode Etik](CODE_OF_CONDUCT.md) proyek ini.

---

Pertanyaan? Buka diskusi di [Issues](https://github.com/Boas04/Sajam-Detector-AI/issues) 💬
