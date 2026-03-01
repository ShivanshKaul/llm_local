import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int
    context_length: int = 128
    embed_dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.embed_dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.embed_dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim)
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(c, dim=2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        if t > self.cfg.context_length:
            raise ValueError(f"Sequence length {t} exceeds context length {self.cfg.context_length}")

        pos = torch.arange(0, t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, s: str):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self):
        return len(self.stoi)


def make_batch(data, batch_size, context_length, device):
    starts = torch.randint(0, len(data) - context_length - 1, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in starts])
    y = torch.stack([data[i + 1 : i + 1 + context_length] for i in starts])
    return x.to(device), y.to(device)


def save_checkpoint(path: Path, model, tokenizer: CharTokenizer, cfg: Config):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "stoi": tokenizer.stoi,
        "cfg": cfg.__dict__,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device):
    payload = torch.load(path, map_location=device)
    cfg = Config(**payload["cfg"])
    tok = CharTokenizer(" ")
    tok.stoi = payload["stoi"]
    tok.itos = {i: ch for ch, i in tok.stoi.items()}
    model = TinyGPT(cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, tok


def train(args):
    text = Path(args.input).read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    split = int(0.9 * len(ids))
    train_data = ids[:split]
    val_data = ids[split:]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg = Config(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        embed_dim=args.embed_dim,
        n_layers=args.layers,
        n_heads=args.heads,
        dropout=args.dropout,
    )
    model = TinyGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        model.train()
        xb, yb = make_batch(train_data, args.batch_size, args.context_length, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vx, vy = make_batch(val_data, args.batch_size, args.context_length, device)
                _, vloss = model(vx, vy)
            print(f"step={step} train_loss={loss.item():.4f} val_loss={vloss.item():.4f}")

    save_checkpoint(Path(args.checkpoint), model, tokenizer, cfg)
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "config": cfg.__dict__,
        "checkpoint": args.checkpoint,
    }
    print("Training complete.")
    print(json.dumps(meta, indent=2))


def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, tokenizer = load_checkpoint(Path(args.checkpoint), device)

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [0]

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.eval()

    for _ in range(args.max_new_tokens):
        x_cond = x[:, -model.cfg.context_length :]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / max(args.temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    print(tokenizer.decode(x[0].tolist()))


def build_parser():
    p = argparse.ArgumentParser(description="Tiny character-level LLM trainer/generator")
    sub = p.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train a tiny model from scratch")
    train_p.add_argument("--input", required=True, help="Path to plain-text training file")
    train_p.add_argument("--checkpoint", default="checkpoints/tiny.pt")
    train_p.add_argument("--steps", type=int, default=1000)
    train_p.add_argument("--batch-size", type=int, default=32)
    train_p.add_argument("--context-length", type=int, default=128)
    train_p.add_argument("--embed-dim", type=int, default=128)
    train_p.add_argument("--layers", type=int, default=4)
    train_p.add_argument("--heads", type=int, default=4)
    train_p.add_argument("--dropout", type=float, default=0.1)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--eval-interval", type=int, default=100)
    train_p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    train_p.set_defaults(func=train)

    gen_p = sub.add_parser("generate", help="Generate text from checkpoint")
    gen_p.add_argument("--checkpoint", required=True)
    gen_p.add_argument("--prompt", default="Hello")
    gen_p.add_argument("--max-new-tokens", type=int, default=200)
    gen_p.add_argument("--temperature", type=float, default=0.9)
    gen_p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    gen_p.set_defaults(func=generate)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
