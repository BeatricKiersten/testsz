# EmpathyTransformer

LLM mini — from scratch. Ringan, cepet, paham emosi.
Train di **Baseten GPU** (SSH), deploy di **Baseten serving**.

## Flow

```
1. Push ke Baseten  → uvx truss train push baseten_train_config.py
2. SSH ke GPU        → ssh training-job-<id>-0.ssh.baseten.co
3. Train model       → python train.py --data data/train_mixed.jsonl ...
4. Download model    → scp dari container
5. Test local        → python inference.py ...
6. Deploy serving    → truss push deploy/baseten/
```

## Arsitektur (43M params, V3)

| Layer | Value | Keterangan |
|-------|-------|------------|
| Vocab size | 16,384 |  |
| d_model | 512 | Embedding size |
| Layers | 12 | Transformer blocks |
| Heads | 8 query → 4 KV | GQA (fused QKV) |
| d_ff | 2,048 (aktual 1365) | SwiGLU FFN |
| Max seq | 512 | Konteks percakapan |
| Thinking | 0-4 loop | Depth-thinking latent |
| **Total** | **~43M** | FP32 ~172MB, INT8 ~43MB |

**Fitur:**
- **KV Cache** — inference O(1) per token, generate cepet
- **Fused QKV** — 1 kernel launch instead of 3
- **Flash Attention** — SDPA 2-3x lebih cepet (CUDA)
- **Depth-thinking loop** — latent recurrent, tanpa token verbal
- **Gradient checkpointing** — hemat VRAM 70%
- RoPE, RMSNorm, SwiGLU, weight tying (dari V1)

## Dataset (37.5K pairs)

Campuran 70% quotes + 30% percakapan natural:

| Sumber | Baris | Konten |
|--------|-------|--------|
| quotes-500k (Kaggle) | 20K | Quotes high english, berbagai emosi |
| english_quotes (HF) | 2.5K | Quotes terkenal, tagged |
| OASST1 (OpenAssistant) | 15K | Percakapan natural Inggris |

Pakai `train_mixed.jsonl` — format `{"input":"...", "response":"...", "emotion":"..."}`

## Training di Baseten (SSH)

```bash
# 1. Install truss & login
uvx truss login
uvx truss ssh setup

# 2. Push training job
uvx truss train push baseten_train_config.py

# 3. Cek job ID
uvx truss train view

# 4. SSH ke GPU instance
ssh training-job-<job_id>-0.ssh.baseten.co

# 5. Di container — transfer code & data
cd $BT_WORKING_DIR
# Dari terminal lain:
scp -r ./data training-job-<id>-0.ssh.baseten.co:$BT_WORKING_DIR/
scp *.py training-job-<id>-0.ssh.baseten.co:$BT_WORKING_DIR/
scp requirements.txt training-job-<id>-0.ssh.baseten.co:$BT_WORKING_DIR/

# 6. Train tokenizer dulu
pip install -r requirements.txt
python tokenizer.py --data ./data/train_mixed.jsonl --vocab-size 16384 --save ./tokenizer.json

# 7. Train model
python train.py --data ./data/train_mixed.jsonl --tokenizer ./tokenizer.json \
                --epochs 5 --batch-size 16 --empathy

# 8. Download model ke local
# Dari terminal lain:
scp training-job-<id>-0.ssh.baseten.co:$BT_WORKING_DIR/checkpoints/best.pt ./checkpoints/
scp training-job-<id>-0.ssh.baseten.co:$BT_WORKING_DIR/tokenizer.json ./
```

## Test inference (local CPU)

```bash
python inference.py --model checkpoints/best.pt --tokenizer tokenizer.json
python inference.py --model checkpoints/best.pt --tokenizer tokenizer.json \
                    --prompt "I feel lonely today"
# Interactive mode
python inference.py --model checkpoints/best.pt --tokenizer tokenizer.json
```

## Deploy ke Baseten serving

```bash
cp checkpoints/best.pt deploy/baseten/model.pt
cp tokenizer.json deploy/baseten/
cd deploy/baseten
truss push
```

## File structure

```
aitest/
├── model.py                 # Arsitektur Transformer
├── config.py                # Hyperparams
├── tokenizer.py             # BPE tokenizer
├── dataset.py               # Dataset loader
├── train.py                 # Training loop (GPU)
├── inference.py             # Inference (CPU/GPU)
├── baseten_train_config.py  # Config push ke Baseten SSH
├── requirements.txt
├── data/
│   ├── train_mixed.jsonl    # 37.5K — quotes + conversations
│   ├── train_quotes.jsonl   # 22.5K — quotes only (text)
│   ├── conversations.jsonl  # 15K   — conv only
│   └── sample.jsonl         # contoh format
├── checkpoints/             # Hasil training
└── deploy/baseten/          # Truss buat serving
    ├── config.yaml
    ├── config.json
    └── model.py
```
