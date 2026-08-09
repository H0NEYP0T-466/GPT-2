import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# ============================================================
# CONFIG - paths, hyperparams
# ============================================================
# On Kaggle, data.txt lives next to this script (/kaggle/working/data.txt).
# When running locally, point DATASET_PATH wherever you like.
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.txt")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pt")
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_log.json")


# ============================================================
# MODEL
# ============================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.bias[:, :, :T, :T], is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Config:
    def __init__(self, block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, target=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_name):
        assert model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        from transformers import GPT2LMHeadModel

        print(f"Loading pre-trained weights: {model_name}")
        config_args = {
            "gpt2": {"n_layer": 12, "n_head": 12, "n_embd": 768},
            "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
            "gpt2-large": {"n_layer": 36, "n_head": 20, "n_embd": 1280},
            "gpt2-xl": {"n_layer": 48, "n_head": 25, "n_embd": 1600},
        }[model_name]
        config_args["block_size"] = 1024
        config_args["vocab_size"] = 50257
        config = GPT2Config(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]
        model_hf = GPT2LMHeadModel.from_pretrained(model_name)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [
            k for k in sd_hf.keys()
            if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias")
        ]
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        assert len(sd_keys) == len(sd_keys_hf), f"Key count mismatch: {len(sd_keys)} vs {len(sd_keys_hf)}"
        for k, k_hf in zip(sd_keys, sd_keys_hf):
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k_hf].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k_hf].t())
            else:
                assert sd_hf[k_hf].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k_hf])
        return model

    def save(self, path=MODEL_SAVE_PATH):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path=MODEL_SAVE_PATH):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict)
        print(f"Model loaded from {path}")

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================
# DATA LOADER
# ============================================================
class DataLoader:
    def __init__(self, B, T, dataset_path, split="train", val_ratio=0.1):
        self.B = B
        self.T = T
        enc = tiktoken.get_encoding("gpt2")

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = f.read()

        tokens = enc.encode(data)
        n = len(tokens)
        split_idx = int(n * (1 - val_ratio))

        if split == "train":
            self.tokens = torch.tensor(tokens[:split_idx])
        else:
            self.tokens = torch.tensor(tokens[split_idx:])

        self.current_position = 0
        print(f"  [{split}] tokens: {len(self.tokens):,} | batches/epoch: {len(self.tokens) // (B * T)}")

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        if len(buf) < B * T + 1:
            self.current_position = 0
            buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0
        return x, y


# ============================================================
# LEARNING RATE SCHEDULE (cosine with warmup)
# ============================================================
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters, device):
    model.eval()
    losses = {}
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        total_loss = 0.0
        for _ in range(eval_iters):
            x, y = loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(x, y)
            total_loss += loss.item()
        losses[split] = total_loss / eval_iters
    model.train()
    return losses


# ============================================================
# GENERATE SAMPLE TEXT
# ============================================================
@torch.no_grad()
def generate_sample(model, enc, device, prompt="Fezan:", max_new_tokens=100):
    model.eval()
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    generated = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.8, top_k=200)
    text = enc.decode(generated[0].tolist())
    model.train()
    return text


# ============================================================
# MAIN TRAINING
# ============================================================
def main():
    # --- Hyperparameters (tuned for ~1.5MB chat dataset on T4) ---
    B = 16                # batch size per step
    T = 512               # context length
    LR = 3e-4             # peak learning rate
    MIN_LR = 3e-5         # min LR (10% of peak)
    WEIGHT_DECAY = 0.1
    GRAD_CLIP = 1.0
    EPOCHS = 20
    WARMUP_RATIO = 0.05   # 5% of total steps for warmup
    VAL_RATIO = 0.1       # 10% validation split
    EVAL_INTERVAL = 50    # evaluate every N steps
    LOG_INTERVAL = 10     # print loss every N steps
    SAMPLE_INTERVAL = 200 # generate sample text every N steps
    GRAD_ACCUM_STEPS = 4  # gradient accumulation (effective batch = 64)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # --- Data ---
    print(f"\nDataset: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: {DATASET_PATH} not found. Place data.txt next to GPT2.py")
        sys.exit(1)

    train_loader = DataLoader(B, T, DATASET_PATH, split="train", val_ratio=VAL_RATIO)
    val_loader = DataLoader(B, T, DATASET_PATH, split="val", val_ratio=VAL_RATIO)

    total_tokens = len(train_loader.tokens) + len(val_loader.tokens)
    print(f"Total tokens: {total_tokens:,}")

    # --- Model ---
    print("\nInitializing GPT-2 (124M)...")
    config = GPT2Config(block_size=T, vocab_size=50257, n_layer=12, n_head=12, n_embd=768)
    model = GPT2(config)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # --- Optimizer ---
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LR, betas=(0.9, 0.95), eps=1e-8)

    # --- Training schedule ---
    tokens_per_step = B * T * GRAD_ACCUM_STEPS
    steps_per_epoch = len(train_loader.tokens) // tokens_per_step
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    print(f"\nTraining config:")
    print(f"  Epochs:            {EPOCHS}")
    print(f"  Batch size:        {B} x {GRAD_ACCUM_STEPS} accum = {B * GRAD_ACCUM_STEPS} effective")
    print(f"  Context length:    {T}")
    print(f"  Steps/epoch:       {steps_per_epoch}")
    print(f"  Total steps:       {total_steps}")
    print(f"  Warmup steps:      {warmup_steps}")
    print(f"  Peak LR:           {LR}")
    print(f"  Min LR:            {MIN_LR}")
    print(f"  Weight decay:      {WEIGHT_DECAY}")
    print(f"  Grad clip:         {GRAD_CLIP}")

    # --- Mixed precision ---
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # --- Logging ---
    log = {
        "config": {
            "B": B, "T": T, "lr": LR, "min_lr": MIN_LR,
            "epochs": EPOCHS, "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP, "grad_accum_steps": GRAD_ACCUM_STEPS,
            "warmup_steps": warmup_steps, "total_steps": total_steps,
            "val_ratio": VAL_RATIO, "dataset_tokens": total_tokens,
        },
        "steps": [],
        "epochs_summary": [],
    }

    # --- Training loop ---
    print(f"\n{'='*60}")
    print("TRAINING START")
    print(f"{'='*60}\n")

    best_val_loss = float("inf")
    model.train()
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        epoch_steps = 0

        for step in range(steps_per_epoch):
            global_step = (epoch - 1) * steps_per_epoch + step + 1

            # --- LR schedule ---
            lr = get_lr(global_step, warmup_steps, total_steps, LR, MIN_LR)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # --- Forward (with grad accumulation) ---
            optimizer.zero_grad()
            loss_accum = 0.0
            for micro_step in range(GRAD_ACCUM_STEPS):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    _, loss = model(x, y)
                    loss = loss / GRAD_ACCUM_STEPS
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_accum += loss.item()

            # --- Gradient clipping ---
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            epoch_loss += loss_accum
            epoch_steps += 1

            # --- Log step ---
            if step % LOG_INTERVAL == 0 or step == steps_per_epoch - 1:
                dt = time.time() - t0
                t0 = time.time()
                tokens_sec = (B * T * GRAD_ACCUM_STEPS) / dt if dt > 0 else 0
                print(
                    f"  epoch {epoch:2d} | step {step+1:4d}/{steps_per_epoch} | "
                    f"loss {loss_accum:.4f} | lr {lr:.2e} | "
                    f"{tokens_sec:.0f} tok/s | {dt:.1f}s"
                )

            # --- Evaluate ---
            if global_step % EVAL_INTERVAL == 0:
                losses = estimate_loss(model, train_loader, val_loader, eval_iters=20, device=device)
                print(f"\n  >> eval @ step {global_step}: train_loss={losses['train']:.4f} | val_loss={losses['val']:.4f}")
                log["steps"].append({
                    "step": global_step,
                    "train_loss": round(losses["train"], 4),
                    "val_loss": round(losses["val"], 4),
                    "lr": round(lr, 8),
                })
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    model.save(MODEL_SAVE_PATH.replace(".pt", "_best.pt"))
                print()

            # --- Generate sample ---
            if global_step % SAMPLE_INTERVAL == 0:
                sample = generate_sample(model, tiktoken.get_encoding("gpt2"), device)
                print(f"\n  >> sample:\n{sample}\n")

        # --- Epoch summary ---
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        losses = estimate_loss(model, train_loader, val_loader, eval_iters=20, device=device)
        epoch_summary = {
            "epoch": epoch,
            "avg_train_loss": round(avg_epoch_loss, 4),
            "val_loss": round(losses["val"], 4),
            "lr": round(lr, 8),
        }
        log["epochs_summary"].append(epoch_summary)
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch} SUMMARY: avg_loss={avg_epoch_loss:.4f} | val_loss={losses['val']:.4f}")
        print(f"{'='*60}\n")

        # Save checkpoint every epoch
        model.save(MODEL_SAVE_PATH)

    # --- Final save ---
    model.save(MODEL_SAVE_PATH)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining log saved to {LOG_PATH}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
