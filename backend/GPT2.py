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
        self.head_size = config.n_embd // config.n_head
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

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
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
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B,T =idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # shape (1, T)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, T, n_embd)  
        x= tok_emb + pos_emb  # (B, T, n_embd)
        for block in self.transformer.h:
            x=block(x)
        x=self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    @classmethod
    def from_pretrained(cls, model_name):
        assert model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], "Model name must be one of: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'"
        from transformers import GPT2LMHeadModel
        print("Loading pre-trained model weights:", model_name)
        config_args={
            'gpt2': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
            'gpt2-medium': {'n_layer': 24, 'n_head': 16, 'n_embd': 1024},
            'gpt2-large': {'n_layer': 36, 'n_head': 20, 'n_embd': 1280},
            'gpt2-xl': {'n_layer': 48, 'n_head': 25, 'n_embd': 1600}
        }[model_name]
        config_args['block_size'] = 1024
        config_args['vocab_size'] = 50257
        # Load the model configuration and weights from a pre-trained model
        config = GPT2Config(**config_args)
        model = cls(config)
        sd=model.state_dict()
        sd_keys=sd.keys()
        sd_keys=[k for k in sd_keys if not k.endswith('.attn.bias')]
        model_hf = GPT2LMHeadModel.from_pretrained(model_name)
        sd_hf=model_hf.state_dict()
        sd_keys_hf=sd_hf.keys()
        sd_keys_hf=[k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        sd_keys_hf=[k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        transposed=['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys)==len(sd_keys_hf), "State dict keys length mismatch"
        for k, k_hf in zip(sd_keys, sd_keys_hf):
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k_hf].shape[::-1]==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k_hf].t())
            else:
                assert sd_hf[k_hf].shape==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k_hf])
        return model


model=GPT2.from_pretrained('gpt2')
print("Model loaded successfully.")