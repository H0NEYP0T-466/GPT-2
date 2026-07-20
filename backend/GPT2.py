from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as f


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
          # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k=k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  #(B, nh, T, hs)
        q=q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  #(B, nh, T, hs)
        v=v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  #(B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))  # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = f.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        y=self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
       
        return x

class block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x

#configurations
@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int=50257
    n_layer: int = 12
    n_head: int =12
    n_embd: int = 768

class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(0.1),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model itself
        token_embeddings = self.transformer.wte(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.transformer.wpe(torch.arange(t, device=device))  # each position maps to a (learnable) vector
        x = self.transformer.drop(token_embeddings + position_embeddings)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # output logits
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # reshape logits and targets for loss computation
            loss = f.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss