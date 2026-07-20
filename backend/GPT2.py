from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as f


#configurations
@dataclass
class DataClass:
    block_size: int = 256
    vocab_size: int=65
    n_layer: int = 8
    n_head: int =6
    n_embd: int = 384

class GPT2(nn.Module):
    def __init__(self, config: DataClass):
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