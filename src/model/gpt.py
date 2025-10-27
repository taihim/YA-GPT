from torch import nn, arange

class VanillaGPT(nn.Module):
    def __init__(self, ctx_len: int = 8, embed_dim: int = 64, vocab_len: int = 65, max_len: int = 256):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_len, embed_dim)
        self.positional_embeddings = nn.Embedding(max_len, embed_dim)


    def forward(self, x) -> None:
        print(x.shape)
        B, T = x.shape

        print(arange(T))
        print(self.positional_embeddings(arange(T)).shape)
        
        return self.embeddings(x) + self.positional_embeddings(arange(T))

    def generate() -> None:
        pass