FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface \
    MEMIR_DB_PATH=/data/memir.db

WORKDIR /app

# Torch CPU-only dulu (tanpa CUDA → image jauh lebih kecil), baru sisanya.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY memir/ ./memir/

# /data = DB (memori), /cache = bobot model. Di-bind ke volume oleh compose.
VOLUME ["/data", "/cache"]

CMD ["python", "-m", "memir"]
