# Studio Proyek 3 — Fine-Tuning Pretrained Transformer (SmSA IndoNLU)

Repositori eksperimen klasifikasi sentimen Bahasa Indonesia menggunakan dataset **SmSA (IndoNLU)** dengan backbone **BERT**, **DistilBERT**, dan **IndoBERT** serta strategi *Full Fine-Tuning*, *Partial Freezing*, dan *Linear Probing*.

Proyek ini menggunakan **[uv](https://docs.astral.sh/uv/)** modern dengan universal lockfile (`uv.lock`) yang mendukung **multi-platform** (Linux, Windows, macOS) serta pilihan backend komputasi (**CPU-only**, **AMD ROCm GPU**, **NVIDIA CUDA GPU**, dan **Apple Silicon MPS**).

---

## Prasyarat

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (`uv >= 0.6`, Python 3.12 dikelola otomatis via `.python-version`).
- *(Khusus AMD ROCm di Linux)* Driver kernel `amdgpu`/KFD aktif. Untuk GPU RDNA2 seperti **AMD Radeon RX 6700 XT** (`gfx1031`), pastikan environment variable berikut aktif:
  ```bash
  export HSA_OVERRIDE_GFX_VERSION=10.3.0
  ```

---

## Instalasi Environment (`uv sync`)

Pilih **satu** backend hardware sesuai perangkat yang digunakan (paket `jupyterlab`, `nbconvert[webpdf]`, `ipykernel`, `ipywidgets`, `ruff`, dll. otomatis ikut terpasang melalui *default dependency groups*):

| Perangkat / Platform | Perintah Instalasi |
| :--- | :--- |
| **AMD ROCm 6.4 GPU** (Linux `x86_64`, direkomendasikan untuk RX 6700 XT / ROCm 6.4) | `uv sync --extra rocm` |
| **CPU-Only** (Linux / Windows) atau **Apple Silicon MPS** (macOS `arm64`) | `uv sync --extra cpu` |
| **NVIDIA CUDA 12.8 GPU** (Linux / Windows) | `uv sync --extra cu128` |

> [!TIP]
> Jika ingin environment ringan tanpa paket dev/linter (`ruff`, `pytest`, `nbstripout`), tambahkan `--no-group dev`:
> ```bash
> uv sync --extra rocm --no-group dev
> ```
> Jika hanya butuh runtime training tanpa JupyterLab/nbconvert:
> ```bash
> uv sync --extra rocm --no-default-groups
> ```

---

## Verifikasi GPU / PyTorch

> [!IMPORTANT]
> Secara default, `uv run <perintah>` akan menjalankan *re-sync* tanpa `--extra` (yang dapat menimpa `torch+rocm` dengan `torch` default dari PyPI). Selalu gunakan `--no-sync` (setelah `uv sync --extra <target>`) atau sertakan `--extra <target>`, atau aktifkan `.venv` terlebih dahulu (`source .venv/bin/activate`).

Cek apakah PyTorch dan akselerator terdeteksi dengan benar:

```bash
uv run --no-sync python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA/ROCm available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('HIP version:', torch.version.hip)
elif torch.backends.mps.is_available():
    print('Device: Apple Silicon MPS')
else:
    print('Device: CPU')
"
```

---

## Menjalankan JupyterLab & Kernel

### 1. Jalankan JupyterLab langsung via `uv`
```bash
uv run --no-sync jupyter lab
# atau: uv run --extra rocm jupyter lab
```

### 2. *(Opsional)* Daftarkan Kernel ke Jupyter / VS Code
```bash
uv run --no-sync python -m ipykernel install --user --name studio3-smsa --display-name "Python 3.12 (Studio3 SmSA)"
```

---

## Ekspor & Eksekusi Notebook dengan `nbconvert`

### Eksekusi notebook secara headless via CLI
```bash
uv run --no-sync jupyter nbconvert --to notebook --execute --inplace studio3_smsa_finetuning.ipynb
```

### Ekspor ke PDF (WebPDF — tanpa perlu LaTeX/TeXLive)
Pertama kali dijalankan, tambahkan `--allow-chromium-download` (atau jalankan `uv run --no-sync playwright install chromium` sekali saja):
```bash
uv run --no-sync jupyter nbconvert --to webpdf --allow-chromium-download \
  --output STUDIO3_KelompokX_Nama1_Nama2.pdf \
  studio3_smsa_finetuning.ipynb
```

### Ekspor ke HTML / LaTeX PDF
```bash
# HTML
uv run --no-sync jupyter nbconvert --to html studio3_smsa_finetuning.ipynb

# LaTeX PDF (membutuhkan pandoc + xelatex)
uv run --no-sync jupyter nbconvert --to pdf studio3_smsa_finetuning.ipynb
```

---

## Utilitas Pengembangan

- **Lint & format (termasuk `.ipynb`)**:
  ```bash
  uv run --no-sync ruff check .
  uv run --no-sync ruff format .
  ```
- **Perbarui lockfile lintas platform** setelah mengubah `pyproject.toml`:
  ```bash
  uv lock
  ```
