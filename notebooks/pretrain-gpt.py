import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset, load_dataset_builder
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt # for making figures
    import tiktoken
    import pyarrow as pa
    import pyarrow.compute as pc
    import os

    return F, tiktoken, torch


@app.cell
def _():
    chunks = []
    with open("TinyStories-train.txt", "r") as f1:
      for line in f1:
        chunks.append(line)

    pretrain_text = "".join(chunks)
    return


@app.cell
def _():
    with open("pg100.txt", "r") as f2:
      shakespeare_text = (f2.read())
    return (shakespeare_text,)


@app.cell
def _(tiktoken):
    encoding = tiktoken.get_encoding("cl100k_base")
    return (encoding,)


@app.cell
def _(encoding, shakespeare_text):
    encoded_text = encoding.encode(shakespeare_text)
    return (encoded_text,)


@app.cell
def _(encoding):
    encoding.n_vocab
    return


@app.cell
def _(encoded_text, torch):
    n = int(0.9*len(encoded_text))
    train_data = torch.tensor(encoded_text[:n])
    val_data = torch.tensor(encoded_text[n:])

    print(f"Training data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")
    return train_data, val_data


@app.cell
def _(encoded_text, encoding):
    encoding.decode(encoded_text[:200])
    return


@app.cell
def _():
    context_size = 16
    batch_size = 4
    return batch_size, context_size


@app.cell
def _(torch, train_data, val_data):
    def get_batch(split="train", context_size=16, batch_size=4):
          data = train_data if split=="train" else val_data
          idx = torch.randint(len(data) - context_size, (batch_size,), )
          x = torch.stack([data[i:i+context_size] for i in idx])
          y = torch.stack([data[i+1:i+context_size+1] for i in idx])
          return x, y

    return (get_batch,)


@app.cell
def _(batch_size, context_size, get_batch, torch):
    def _():
        torch.manual_seed(1337)

        xb, yb = get_batch()
        print(xb.shape)
        print(yb.shape)
        print(xb[0])
        print(yb[0])

        print("-----" * 15)
        print("")

        for i in range(batch_size):
          for l in range (context_size):
            print(f"When input is {xb[i, :l+1].tolist()}, target is {yb[i, l]}")
        return

    _()
    return


@app.cell
def _(F, encoding, torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Combine the `Head` and `MultiHeadAttention` into one class that processes all the heads in parallel,
    # treating the heads as another batch dimension


    # 64, 256, 512
    # 2 head attention -> 64, 256, 256
    # after concat -> 64, 256, 512

    # how to do this in parallel?
    # treat each head as a batch dimension
    # if 2 heads
    # 64, 2, 256, 256?
    # then combine?

    # q,k,v wont be small size then i.e. they will be embd_size, embd_size instead of embd_size, embd_size // num_heads
    # first we reshape from B, T, C to B, T, num_heads, embd_size // num_heads (i.e. head_size)
    # then we transpose num_heads and T dimension to get B, num_heads, T, head_size
    # we carry out all operations on this matrix
    # at the end we reshape back to B, T, C

    import math
    # import torch
    import torch.nn as nn
    torch.manual_seed(1337)

    embd_size = 512
    max_len = 256
    block_size = 512
    num_blocks = 8
    dropout = 0.2

    class MultiHeadAttention(nn.Module):
      def __init__(self, n_embd, n_heads):
        super().__init__()
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_size = n_embd // n_heads

        self.Wk = nn.Linear(n_embd, n_embd)
        self.Wq = nn.Linear(n_embd, n_embd)
        self.Wv = nn.Linear(n_embd, n_embd)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

      def forward(self, x):
        B, T, C = x.shape

        k = self.Wk(x)
        q = self.Wq(x)
        v = self.Wv(x)

        k = k.view(B, T, self.n_heads, self.head_size)
        q = q.view(B, T, self.n_heads, self.head_size)
        v = v.view(B, T, self.n_heads, self.head_size)

        k = k.transpose(2, 1)
        q = q.transpose(2, 1)
        v = v.transpose(2, 1)


        wei = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v

        out = out.transpose(1, 2).reshape(B, T, C)

        out = self.dropout(self.proj(out))

        return out

    class Ffwd(nn.Module):
      def __init__(self, embd_size, head_size, n_heads):
        super().__init__()
        # self.ffwd = nn.Sequential(nn.Linear(embd_size, head_size * n_heads), nn.ReLU(), nn.Linear(head_size * n_heads, embd_size), nn.Dropout(dropout))
        # gemini suggested expanding linear layer 512 * 4 = 2048
        # self.ffwd = nn.Sequential(nn.Linear(embd_size, 4 * embd_size), nn.ReLU(), nn.Linear(4 * embd_size, embd_size), nn.Dropout(dropout))
        self.ffwd = nn.Sequential(nn.Linear(embd_size, 4 * embd_size), nn.GELU(), nn.Linear(4 * embd_size, embd_size), nn.Dropout(dropout))

      def forward(self, x):
        return self.ffwd(x)


    class Block(nn.Module):
      def __init__(self, embd_size, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embd_size)
        self.mh_attn = MultiHeadAttention(embd_size, n_heads)
        self.ln2 = nn.LayerNorm(embd_size)
        self.ffwd = Ffwd(embd_size, embd_size // n_heads, n_heads)

      def forward(self, x):
        out = x + self.mh_attn(self.ln1(x))
        out = out + self.ffwd(self.ln2(out))

        return out


    class AttentionLanguageModel(nn.Module):
      def __init__(self, vocab_size, n_heads, num_blocks):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, embd_size)
        self.position_embedding_table = nn.Embedding(max_len, embd_size)

        self.blocks = nn.ModuleList([Block(embd_size, n_heads) for _ in range(num_blocks)])

        self.ff = nn.Linear(embd_size, vocab_size)


      def forward(self, idx, target=None):
        _, T = idx.shape
        logits = self.embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=idx.device))

        out = logits
        for block in self.blocks:
          out = block(out)

        out = self.ff(out)

        loss = None
        if not target is None:
          B,T,C = out.shape
          out = out.view(B*T, C)

          targets = target.view(B*T)
          loss = F.cross_entropy(out, targets)

        return out, loss

      def generate(self, ctx, max_tokens):
        for _ in range(max_tokens):
          logits, _ = self(ctx)

          # get only last timestep becuase thats the prediction for whats next
          # we already know the preceding chars

          logits = logits[:, -1, :]

          probs = F.softmax(logits, dim=1)
          generated_token = torch.multinomial(probs, num_samples=1)
          ctx = torch.cat((ctx, generated_token), dim=1)

        return ctx

    # m = AttentionLanguageModel(len(pretrain_vocab), 8, num_blocks).to(device)
    m = AttentionLanguageModel(encoding.n_vocab, 8, num_blocks).to(device)

    # xb = xb.to(device)
    # output, _ = m(xb)
    # print(output.shape)
    return device, m


@app.cell
def _(get_batch, m, torch):
    for p in m.parameters():
      print(p.shape)

    optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)
    eval_iters = 200
    def estimate_loss(model):
        device = next(model.parameters()).device
        model.eval()
        losses = {'train': 0.0, 'val': 0.0}
        for split in ['train', 'val']:
            total_loss = 0.0
            for _ in range(eval_iters):
                xb, yb = get_batch(split)
                xb = xb.to(device)
                yb = yb.to(device)
                _, loss = model(xb, yb)
                total_loss += loss.item()
            losses[split] = total_loss / eval_iters
        model.train()
        return losses

    return estimate_loss, optimizer


@app.cell
def _(device, estimate_loss, get_batch, m, optimizer):
    # parallel multi head attn (4 heads, 128 head size, 512 embedding)
    # more ram usage but faster training
    # 8 epochs in 30mins instead of 7
    # 12gb ram instead of 7.5gb
    # batch_size = 32
    epochs = 5000
    eval_interval = 500

    for i in range(epochs):
      xb, yb = get_batch('train', 16, 16)

      xb = xb.to(device)
      yb = yb.to(device)

      if i % eval_interval == 0:
        losses = estimate_loss(m)
        print(f"Train loss {losses['train']} and eval loss: {losses['val']}")

      logits, loss = m(xb, yb)
      optimizer.zero_grad(set_to_none=True)

      loss.backward()

      optimizer.step()

    print(loss)
    return


@app.cell
def _(device, encoding, m, torch):
    # pretraining gpt generation with tiktoken encodings 2 runs 5.32 train loss 5.62 val loss

    gen_start = torch.zeros((1, 1), dtype=torch.long).to(device)
    output = "".join([encoding.decode([idx]) for idx in m.generate(gen_start, max_tokens=256).squeeze().tolist()])
    print(output)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
