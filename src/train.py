from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_heads: int = 12
    n_embd: int = 768

# @dataclass
# class GPTConfig:
#     block_size: int = 512
#     vocab_size: int = 50257
#     n_layer: int = 6
#     n_heads: int = 6
#     n_embd: int = 384


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) 

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size()

        kqv = self.c_attn(x)
        q, k, v = kqv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)
        y = att @ v
        y = y.transpose(2, 1).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y



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


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, model_type):
        from transformers import GPT2LMHeadModel

        config_args = {"gpt2": dict(n_layer=12, n_heads=12, n_embd=768)}["gpt2"] # 124M params

        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()

        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()

        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape == sd[k].shape[::-1], f"transposed shape mismatch for {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape, f"shape mismatch for {k}: {sd_hf[k].shape} != {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)

        tok_emb = self.transformer.wte(idx)
        
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x) #(B, T, vocab_size)

        return logits

    def generate(self, ctx):
        print(ctx)
        # ctx will be a 1, 1 tensor
        # we will pass this to the model and get a 1, 1, 384 tensor that will then go on to give 1, 1, 50257 logits
        # we will then do softmax on the logits and pick the highest probability. or use torch.multinomial
        logits = self.forward(ctx)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        output = torch.multinomial(topk_probs, num_samples=1)

        xcol = torch.gather(topk_indices, -1, output)
        # output = output

        return torch.cat((ctx, xcol), dim=1)


def generate_sentence(m, seed_str):
    pass


import tiktoken
enc = tiktoken.get_encoding("gpt2")

if __name__ == "__main__":
    m = GPT.from_pretrained("gpt-2")
    torch.manual_seed(42)
    inp = torch.tensor(enc.encode("Hello, I'm a language model."), dtype=torch.int).view(1, -1)
    # print(torch.tensor(inp).view(1, -1))
    max_len = 30
    while inp.size(1) < max_len:
        inp = m.generate(inp)
    
    print(enc.decode(inp.tolist()[-1]))
    # print(torch.ones((1, 1)).size())
    # print("didn't crash, wooo")