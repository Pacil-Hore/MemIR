# MemIR

Discord bot untuk **semantic memory/search**: cari pesan berdasarkan makna, bukan
keyword. Cari "deadline" tetap nemu "tenggat waktu" / "due date".

Desain lengkap & keputusan: [docs/MemIR-plan.md](docs/MemIR-plan.md).

## Struktur

```
memir/
├── models.py      # MessageRecord, SearchResult (tipe domain)
├── config.py      # konstanta kalibrasi (THRESHOLD, TOP_K, model, prefix)
├── embedder.py    # Embedder (ABC) + E5Embedder — prefix e5 + normalisasi
├── store.py       # MemoryStore (ABC) + SqliteStore — BLOB + NumPy in-memory
├── services.py    # IngestService, RecallService (logika domain)
├── bot.py         # lapisan Discord (on_message, !mem, !reindex)
└── __main__.py    # composition root
```

Abstraksi `Embedder` & `MemoryStore` adalah seam upgrade: naik ke e5-base/large
atau pindah ke FAISS/Qdrant tanpa menyentuh bot/service.

## Setup

```bash
pip install -r requirements.txt        # narik torch — agak besar
cp .env.example .env                    # isi DISCORD_TOKEN
```

Di **Discord Developer Portal → Bot**, aktifkan **Message Content Intent**
(tanpa ini `msg.content` selalu kosong). Invite bot dengan izin baca pesan +
**Read Message History** (untuk `!reindex`).

## Jalankan (lokal)

```bash
python -m memir
```

## Deploy di VPS (Docker, 24/7)

`restart: unless-stopped` bikin bot otomatis nyala lagi kalau crash atau VPS
reboot. Volume bikin DB (memori) & bobot model tahan redeploy.

**Setup sekali di VPS** (sudah ada Docker + Docker Compose):

```bash
git clone <repo-url> memir && cd memir
cp .env.example .env && nano .env     # isi DISCORD_TOKEN
docker compose up -d --build          # build pertama agak lama (torch + model)
docker compose logs -f                # cek "MemIR siap sebagai ..."
```

Perintah harian: `docker compose up -d --build` (deploy), `docker compose down`
(stop), `docker compose logs -f` (log).

## CI/CD (GitHub Actions)

`.github/workflows/ci-cd.yml`:

- **CI** (tiap push & PR): compile + `pytest`. Pakai FakeEmbedder → **tanpa
  torch**, jadi cepat.
- **CD** (push ke `main`, setelah CI lulus): SSH ke VPS → `git pull` +
  `docker compose up -d --build`.

Set secret repo di **Settings → Secrets and variables → Actions**:

| Secret | Isi |
|---|---|
| `VPS_HOST` | IP / hostname VPS |
| `VPS_USER` | user SSH (mis. `deploy`) |
| `VPS_SSH_KEY` | private key SSH (yang publik-nya ada di `authorized_keys` VPS) |
| `VPS_APP_DIR` | path repo di VPS (mis. `/home/deploy/memir`) |

Token bot **tidak** masuk GitHub secrets — cukup ada di `.env` pada VPS.

### Sizing VPS
~1–2 GB RAM (torch + e5-small di CPU) dan ~3–4 GB disk (image + bobot model).

## Command

| Command | Fungsi |
|---|---|
| `!mem <query>` | Cari di seluruh pesan server |
| `!mem <query> @user` | Cari, difilter pesan author tertentu |
| `!reindex` | Backfill pesan history channel (idempotent) |

## Kalibrasi

`THRESHOLD` di [memir/config.py](memir/config.py) adalah tuas utama. e5 +
embedding ternormalisasi kasih cosine tinggi & rapat, jadi mulai di `0.80`:
naikkan kalau hasil ngawur lolos, turunkan kalau yang relevan kebuang.
