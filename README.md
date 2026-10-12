# MiniLab AI & Machine Learning — versi pengembangan UAS

Aplikasi Streamlit ini mengembangkan fitur berdasarkan arahan dosen:

- Regresi Linear, klasifikasi, dan clustering K-Means.
- Impor dataset langsung dari UCI Machine Learning Repository menggunakan `ucimlrepo`.
- Upload CSV, simpan metadata dan baris data ke MongoDB, lalu retraining model dari data MongoDB.
- Dataset cuaca real-time/prakiraan terbaru melalui Open-Meteo API, dengan opsi menyimpan data ke MongoDB.
- Demonstrasi neural network sederhana untuk klasifikasi Iris dengan TensorFlow.
- Komentar publik tersimpan di MongoDB, beserta ringkasan otomatis dan klasifikasi sentimen melalui Groq; Gemini tersedia sebagai pilihan tambahan.

## 1. Persyaratan

Disarankan Python 3.11 (lihat `.python-version`). Pastikan jaringan internet aktif untuk mengambil dataset UCI, cuaca, dan menghubungi layanan LLM.

Windows / PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

TensorFlow dan semua paket dipasang dari `requirements.txt`. Proses instalasi pertama kali dapat cukup lama karena TensorFlow berukuran besar.

## 2. Konfigurasi lokal

Buat folder `.streamlit` dan file `.streamlit/secrets.toml` (jangan commit ke GitHub):

```toml
MONGODB_URI = "mongodb+srv://USERNAME:PASSWORD@CLUSTER/"
GROQ_API_KEY = "masukkan_api_key_groq"
GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_API_KEY = "masukkan_api_key_gemini"
GEMINI_MODEL = "gemini-3.8-flash"
```

- `MONGODB_URI` diperlukan untuk menyimpan dataset, komentar, dan data cuaca.
- `GROQ_API_KEY` diperlukan untuk auto-summary dan analisis sentimen utama.
- `GEMINI_API_KEY` opsional, jika ingin menguji alternatif Gemini.
- Simpan secret hanya di file lokal atau panel Secrets pada platform hosting. Jangan unggah secret ke repository publik.

Buat akun MongoDB Atlas, cluster, database user, dan aturan Network Access yang sesuai. Nama database yang digunakan aplikasi adalah `uas_streamlit`.

## 3. Cara menguji fitur

1. **Demo Model ML**: lihat regresi linear, klasifikasi Iris, serta K-Means.
2. **Dataset & Retraining**: pilih CSV atau UCI Repository. Contoh ID UCI: `53` (Iris), `17` (Breast Cancer Wisconsin Original), atau `45` (Heart Disease). Setelah data muncul, tentukan jenis model, fitur, target bila diperlukan, lalu simpan ke MongoDB. Pilih dataset tersimpan dan klik retraining.
3. **Data Real-Time**: ambil prakiraan cuaca untuk koordinat yang dipilih. Data ditampilkan dalam bentuk metrik/grafik, dapat diunduh sebagai CSV, atau disimpan ke MongoDB.
4. **Deep Learning**: tekan tombol latihan untuk melatih dense neural network sederhana pada dataset Iris, kemudian lihat accuracy, loss, dan kurva pelatihan.
5. **Komentar Publik + AI**: kirim beberapa komentar. Pilih Groq lalu jalankan analisis untuk merangkum sampai 30 komentar terbaru dan memberikan label positif/negatif/netral yang disimpan kembali ke MongoDB. Gemini dapat dipilih bila API key terkonfigurasi.

## 4. Menjalankan di Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `README.md`, dan `.python-version` ke GitHub. Jangan push file `secrets.toml`.
2. Deploy repository di Streamlit Community Cloud.
3. Masukkan secrets dalam pengaturan aplikasi, dengan format TOML seperti pada contoh di atas.
4. Pastikan MongoDB Atlas mengizinkan koneksi dari lingkungan hosting.
5. Uji setiap halaman sesudah deploy sebelum mengirimkan URL kepada dosen.

## Catatan batasan

- Open-Meteo menyediakan data cuaca dan prakiraan yang diperbarui berkala; ini bukan sensor langsung di rumah/lokasi pengguna. Penggunaan API harus mengikuti ketentuan penyedia.
- Dataset UCI memiliki format beragam. Pada fitur retraining versi ini, kolom fitur yang dipilih harus numerik. Untuk dataset berkategori, pilih fitur numerik dan target yang sesuai.
- Analisis sentimen dari Groq/Gemini merupakan prediksi LLM dan dapat salah. Periksa hasil secara manual bila digunakan untuk penilaian penting.
- Model TensorFlow didemonstrasikan pada data Iris dan dilatih ulang ketika tombol ditekan; ini bukan sistem produksi.
- Aplikasi memerlukan konfigurasi layanan eksternal agar seluruh fitur berfungsi. Sintaks kode dapat diperiksa lokal, namun akses MongoDB/API dan deployment harus diuji menggunakan kredensial pengguna sendiri.
