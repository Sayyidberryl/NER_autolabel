# Gunakan base image Python yang resmi, ramping (slim), dan aman untuk production
FROM python:3.10-slim

# Set environment variables untuk mengoptimalkan Python di dalam container:
# - PYTHONDONTWRITEBYTECODE=1: Mencegah Python menulis file .pyc (bytecode) ke dalam disk
# - PYTHONUNBUFFERED=1: Memastikan output log Python (stdout & stderr) langsung dicetak ke terminal secara real-time tanpa di-buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Tentukan direktori kerja (working directory) utama di dalam container
WORKDIR /app

# Buat user non-root khusus bernama 'appuser' dan berikan kepemilikan direktori /app kepadanya demi keamanan
RUN useradd --create-home appuser && chown -R appuser /app

# Salin file dependensi (requipment.txt) ke dalam working directory
# Kita menggunakan --chown agar file langsung dimiliki oleh 'appuser'
COPY --chown=appuser:appuser requipment.txt .

# Install semua library Python yang dibutuhkan tanpa menyimpan cache instalasi pip untuk memperkecil ukuran image
RUN pip install --no-cache-dir -r requipment.txt

# Salin file script utama (ner_autolabel.py) ke dalam working directory
COPY --chown=appuser:appuser ner_autolabel.py .

# Ganti user aktif container ke user non-root yang sudah dibuat sebelumnya demi keamanan runtime
USER appuser

# Tentukan perintah default untuk menjalankan script Python ketika container dijalankan
CMD ["python", "ner_autolabel.py"]
