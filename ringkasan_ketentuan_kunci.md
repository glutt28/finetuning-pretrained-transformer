# Ringkasan Ketentuan Kunci & Rekomendasi Konfigurasi Eksperimen
**Mata Kuliah:** Pembelajaran Mendalam (IF25-40401) — Studio Proyek 3  
**Topik:** Fine-Tuning Transformer Pretrained untuk NLP (Klasifikasi Sentimen SmSA)  
**Kelompok:** 2 Anggota (Peran 1: Pipeline, BERT, DistilBERT, Konsolidasi; Peran 2: IndoBERT)  

Dokumen ini memuat seluruh ringkasan aturan resmi dari `Spesifikasi_Tugas_Studio_Proyek_3.pdf` serta rekomendasi konfigurasi teknis yang telah diisi lengkap. 
Setiap poin telah ditandai statusnya:
- 🔴 **`[WAJIB SPESIFIKASI]`**: Aturan baku yang tidak boleh diubah/dilanggar agar tidak gugur penilaian.
- 🟢 **`[REKOMENDASI - BISA DIGANTI]`**: Nilai/desain usulan optimal yang bisa disesuaikan kembali berdasarkan kesepakatan kelompok atau keterbatasan komputasi.

---

## 1. Spesifikasi Tugas & Dataset

| Parameter | Konfigurasi Terpilih / Rekomendasi | Status Ketentuan | Catatan & Alasan |
| :--- | :--- | :--- | :--- |
| **Domain Masalah** | Klasifikasi Sentimen Teks Bahasa Indonesia (3 Kelas: *Negatif*, *Netral*, *Positif*) | 🔴 `[WAJIB SPESIFIKASI]` | Mengacu pada benchmark resmi IndoNLU (Wilie dkk., 2020). |
| **Pilihan Dataset** | **SmSA** (*IndoNLU*) | 🔴 `[WAJIB SPESIFIKASI]` | Pilihan dataset utama Bahasa Indonesia. |
| **Aturan Pembersihan** | 1. Hapus duplikasi baris teks.<br>2. Hapus baris dengan *label contradiction* (teks sama label beda).<br>3. Hapus data validasi yang bocor ke data latih. | 🔴 `[WAJIB SPESIFIKASI]` | Mekanisme bawaan notebook untuk memastikan evaluasi tidak bias. |
| **Pembagian Data** | Split resmi bawaan SmSA (*Train*, *Validation*, *Test*) | 🔴 `[WAJIB SPESIFIKASI]` | Menggunakan split benchmark baku. |
| **Ukuran Subset** | **100% (Seluruh Data SmSA)** | 🟢 `[REKOMENDASI - BISA DIGANTI]` | SmSA relatif ringkas (~11.000 train). Menggunakan seluruh data memberikan performa stabil. Jika kuota GPU sangat mepet, bisa dikurangi (misal: 5.000 sampel latih) asalkan ada justifikasi tertulis di sel Rancangan. |

---

## 2. Pemilihan Model (Backbone)

| Model / Backbone | HuggingFace Identifier | Peran / Pembagian | Status Ketentuan |
| :--- | :--- | :--- | :--- |
| **BERT** | `bert-base-multilingual-cased` | **Anggota 1** (Baseline pembanding multibahasa) | 🔴 `[WAJIB SPESIFIKASI]` |
| **DistilBERT** | `distilbert-base-multilingual-cased` | **Anggota 1** (Analisis efisiensi komputasi vs akurasi) | 🔴 `[WAJIB SPESIFIKASI]` |
| **IndoBERT** | `indobenchmark/indobert-base-p1` | **Anggota 2** (Model terspesialisasi Bahasa Indonesia) | 🔴 `[WAJIB SPESIFIKASI]` (Wajib jika dataset SmSA) |
| **Model English-only** | `bert-base-uncased`, `roberta-base`, dll. | **DILARANG DIGUNAKAN PADA SmSA** | 🔴 `[WAJIB SPESIFIKASI]` (Larangan keras) |
| **Model Tambahan (Opsional)** | `xlm-roberta-base` | Nilai plus (jika waktu komputasi mencukupi) | 🟢 `[REKOMENDASI - BISA DIGANTI]` (Opsional, diabaikan dulu untuk efisiensi) |

---

## 3. Strategi Fine-Tuning & Pembekuan Lapisan (Freezing)

| Strategi | Definisi & Implementasi Teknis | Status Ketentuan |
| :--- | :--- | :--- |
| **Head Replacement** | Mengganti *classification head* bawaan sesuai jumlah kelas SmSA (3 kelas logits). | 🔴 `[WAJIB SPESIFIKASI]` (Wajib di semua run) |
| **Full Fine-Tuning** | Semua bobot parameter (backbone + head) di-update selama pelatihan. | 🔴 `[WAJIB SPESIFIKASI]` (Wajib sebagai baseline) |
| **Partial Freezing** | **Aturan Pembekuan:**<br>- Bekukan seluruh layer *Embeddings*.<br>- Bekukan **separuh awal transformer encoder layers**:<br>  * BERT (12 layer) $\rightarrow$ Layer 0–5 beku, Layer 6–11 dilatih.<br>  * DistilBERT (6 layer) $\rightarrow$ Layer 0–2 beku, Layer 3–5 dilatih.<br>  * IndoBERT (12 layer) $\rightarrow$ Layer 0–5 beku, Layer 6–11 dilatih. | 🔴 `[WAJIB SPESIFIKASI]` |
| **Linear Probing** | Bekukan seluruh layer backbone (Embeddings + seluruh Encoder); **hanya classifier head** yang dilatih. | 🟢 `[REKOMENDASI - BISA DIGANTI]` (Minimal pilih salah satu antara Linear Probing atau Partial Freezing; direkomendasikan menjalankan keduanya jika kuota cukup). |
| **Verifikasi Parameter** | Jumlah parameter terlatih harus terbukti secara kode memenuhi urutan ketat:<br>$$\mathbf{N_{params}(Probing) < N_{params}(Partial) < N_{params}(Full)}$$ | 🔴 `[WAJIB SPESIFIKASI]` |

---

## 4. Hyperparameter & Variabel Terkendali

Semua nilai di bawah wajib dikunci dan disamakan persis antara Anggota 1 dan Anggota 2 sebelum eksperimen dimulai.

| Variabel | Nilai Rekomendasi | Rentang Batasan Spesifikasi | Status Ketentuan |
| :--- | :--- | :--- | :--- |
| **Random Seeds** | `[42, 123, 2026]` | Minimal 3 seed acak identik | 🟢 `[REKOMENDASI - BISA DIGANTI]` (Angkanya bebas, tapi wajib minimal 3 seed dan harus identik di semua model) |
| **Max Sequence Length** | `128` | Disesuaikan dengan panjang kalimat SmSA | 🟢 `[REKOMENDASI - BISA DIGANTI]` (Dapat diubah ke 96 atau 128; 128 mencakup >98% distribusi SmSA) |
| **Batch Size** | `32` (Fallback: `16` jika OOM) | Disesuaikan dengan VRAM T4 | 🟢 `[REKOMENDASI - BISA DIGANTI]` |
| **Epochs (Full & Partial)** | `3` epoch | Bebas ditentukan kelompok | 🟢 `[REKOMENDASI - BISA DIGANTI]` (Umumnya konvergen di 3–4 epoch) |
| **Epochs (Linear Probing)** | `5` epoch | Bebas ditentukan kelompok | 🟢 `[REKOMENDASI - BISA DIGANTI]` (Probing butuh epoch lebih banyak karena hanya melatih head) |
| **Learning Rate (Full)** | `3e-5` ($3 \times 10^{-5}$) | **Wajib dalam rentang:** $2\times 10^{-5} - 5\times 10^{-5}$ | 🔴 `[WAJIB SPESIFIKASI: RENTANG]`<br>🟢 `[REKOMENDASI: NILAI 3e-5]` |
| **Learning Rate (Partial)** | `3e-5` ($3 \times 10^{-5}$) | **Wajib dalam rentang:** $2\times 10^{-5} - 5\times 10^{-5}$ | 🔴 `[WAJIB SPESIFIKASI: RENTANG]`<br>🟢 `[REKOMENDASI: NILAI 3e-5]` |
| **Learning Rate (Probing)** | `1e-3` ($1 \times 10^{-3}$) | **Wajib dalam rentang:** $1\times 10^{-3} - 5\times 10^{-3}$ | 🔴 `[WAJIB SPESIFIKASI: RENTANG]`<br>🟢 `[REKOMENDASI: NILAI 1e-3]` |
| **Optimizer** | `AdamW` (Weight decay = `0.01`) | Standar transformer | 🟢 `[REKOMENDASI - BISA DIGANTI]` |
| **Warmup Ratio** | `0.1` (10% total training steps) | Standar linear schedule | 🟢 `[REKOMENDASI - BISA DIGANTI]` |

---

## 5. Matriks Eksperimen & Pembagian Beban Kerja

### Opsi A: Matriks Penuh ($3 \times 3 \times 3 = 27$ Run) — *(Sangat Direkomendasikan)*
Jika kuota GPU Google Colab / Kaggle mencukupi (total waktu latih sekitar 60–90 menit dibagi 2 akun):

- **Anggota 1 (Anda):**
  - BERT $\times$ [Full, Partial, Probing] $\times$ 3 Seeds = **9 Run**
  - DistilBERT $\times$ [Full, Partial, Probing] $\times$ 3 Seeds = **9 Run**
  - *Total:* **18 Run** (~45–60 menit GPU T4)
- **Anggota 2:**
  - IndoBERT $\times$ [Full, Partial, Probing] $\times$ 3 Seeds = **9 Run** (~25–30 menit GPU T4)

### Opsi B: Matriks Minimum ($3 \times 2 \times 3 = 18$ Run) — *(Alternatif Hemat Waktu/Kuota)*
Jika kuota komputasi sangat terbatas, spesifikasi mengizinkan memilih salah satu antara Linear Probing atau Partial Freezing (disandingkan dengan Full Fine-Tuning):
- Strategi yang diuji: **Full Fine-Tuning vs Partial Freezing**
- Anggota 1: 2 backbone $\times$ 2 strategi $\times$ 3 seed = **12 Run**
- Anggota 2: 1 backbone $\times$ 2 strategi $\times$ 3 seed = **6 Run**

*Status:* 🟢 `[REKOMENDASI - BISA DIGANTI]` (Boleh pilih Opsi A atau Opsi B).

---

## 6. Protokol Evaluasi & Konsolidasi Inferensi

1. **Metrik Evaluasi Per Run:**
   - Akurasi (*Accuracy*)
   - F1-Score Makro (*Macro F1*)
   - Waktu Latih (*Training Time* - detik)
   - Waktu Inferensi (*Latency* - `ms/kalimat`)
   - Parameter Total & Parameter Terlatih (*Trainable Params*)
   - Memori Puncak GPU (*Peak VRAM Usage* - MB)
   *(Status: 🔴 `[WAJIB SPESIFIKASI]`)*

2. **Aturan Uji Kecepatan Inferensi:**
   - 🔴 `[WAJIB SPESIFIKASI]`: **Waktu inferensi (`ms/kalimat`) hanya valid jika diuji dalam SATU SESI PERANGKAT YANG SAMA.**
   - **Mekanisme Konsolidasi:** Anggota 2 mengekspor/menyimpan model checkpoint IndoBERT (atau state dict-nya), lalu Anggota 1 memuat seluruh model (BERT, DistilBERT, IndoBERT) ke dalam satu runtime GPU yang sama untuk menjalankan benchmark inferensi pada data test SmSA.

3. **Format Pelaporan Hasil:**
   - 🔴 `[WAJIB SPESIFIKASI]`: Pelaporan akhir wajib berupa **Rata-rata $\pm$ Simpangan Baku** ($\text{Mean} \pm \text{Std}$) dari ketiga seed. Dilarang hanya melaporkan skor tertinggi (*cherry-picking*).

---

## 7. Format Berkas Luaran & Pengumpulan

| Berkas Luaran | Format / Penamaan Baku | Status Ketentuan |
| :--- | :--- | :--- |
| **Tautan Notebook** | Tautan aktif Google Colab / Kaggle Notebook | 🔴 `[WAJIB SPESIFIKASI]` |
| **PDF Notebook Lengkap** | `STUDIO3_Kelompok[X]_[NamaAnggota1]_[NamaAnggota2].pdf` | 🔴 `[WAJIB SPESIFIKASI]` |
| **Berkas Raw Run** | `hasil_<dataset>_<kode>.csv` *(dihasilkan otomatis oleh notebook)* | 🔴 `[WAJIB SPESIFIKASI]` |
| **Berkas Agregat** | `rekap_agregat_<dataset>_<kode>.csv` *(rata-rata $\pm$ simpangan baku)* | 🔴 `[WAJIB SPESIFIKASI]` |
| **Slide & Laporan** | Slide presentasi (maks 15 slide) + Laporan ringkas (3–4 halaman) | 🔴 `[WAJIB SPESIFIKASI]` |
| **Batas Pengumpulan** | **Paling lambat 20 November 2026** | 🔴 `[WAJIB SPESIFIKASI]` |
