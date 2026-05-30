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

## Jalankan

```bash
python -m memir
```

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
