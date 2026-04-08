# Tiny Local LLM Starter (From Scratch)

This repo is a beginner-friendly starter to train a **very small character-level Transformer** on local text data.

## What you can do with 8 GB RAM

- ✅ Learn core LLM concepts (tokenization, self-attention, training loop, generation).
- ✅ Train tiny models (a few million parameters) on CPU or small GPU.
- ⚠️ Not enough for training production-grade chat models from scratch.

## Requirements

- Python 3.10+
- PyTorch 2.x
- 2-8 GB free disk (for data + checkpoints)

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

1) Add text to `data/input.txt`.

2) Train:

```bash
python tiny_llm.py train \
  --input data/input.txt \
  --steps 1000 \
  --batch-size 32 \
  --context-length 128 \
  --embed-dim 128 \
  --layers 4 \
  --heads 4 \
  --checkpoint checkpoints/tiny.pt
```

3) Generate:

```bash
python tiny_llm.py generate \
  --checkpoint checkpoints/tiny.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 200
```

4) Graphify loss curves from the checkpoint:

```bash
python tiny_llm.py graphify \
  --checkpoint checkpoints/tiny.pt \
  --output artifacts/loss_curve.svg \
  --title "TinyGPT train vs val loss"
```

This uses training history saved during `train` and writes an SVG loss curve.

## Suggested defaults for 8 GB RAM

- `--batch-size 16` or `32`
- `--context-length 64` to `128`
- `--embed-dim 96` to `192`
- `--layers 2` to `4`

If memory errors occur, reduce batch size first, then context length.

## What to replace `data/input.txt` with

Use a **plain UTF-8 text corpus** that matches the writing style you want the tiny model to imitate.

Good beginner choices:
- Public-domain books (Project Gutenberg `.txt`)
- Your own notes, blogs, or documentation exports
- A directory of `.txt` files merged into one file

Recommended size for first runs:
- Minimum: ~200 KB
- Better: 1-20 MB

Avoid:
- Binary files (`.pdf`, `.docx`, images) pasted directly
- Very tiny files (model will just memorize and repeat)
- Mixed encodings (save as UTF-8)

Example to build `data/input.txt` from multiple text files:

```bash
cat my_corpus/*.txt > data/input.txt
```

Quick cleanup ideas (optional):

```bash
# Remove blank lines
awk 'NF' data/input.txt > data/input.clean.txt
mv data/input.clean.txt data/input.txt
```

Tip: since this project uses **character-level tokenization**, punctuation and spacing patterns in your corpus strongly affect output style.
