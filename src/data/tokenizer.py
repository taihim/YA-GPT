import torch
import tiktoken

class SimpleTokenizer:
    """Basic tokenizer for character level encoding."""
    def __init__(self, text: str) -> None:
        self.vocab = sorted(list(set(text)))    

        self.ctoi = {c: idx for idx, c in enumerate(self.vocab)}
        self.itoc = {idx: c for idx, c in enumerate(self.vocab)}

        print(len(self.vocab))
        print(self.vocab) 
        print(self.ctoi)
        print(self.itoc)

        self.encode = lambda string: torch.tensor([self.ctoi[char] for char in string])
        self.decode = lambda ints: "".join([self.itoc[i] for i in ints])
        
if __name__ == "__main__":
    enc = tiktoken.get_encoding("cl100k_base")
    print(f"Vocab size: {enc.n_vocab}")

    for i in range(100):
        print(f"Token {i+100}: {repr(enc.decode([i+1000]))}")

    text = "Hello, world!"
    tokens = enc.encode(text)
    print(f"\n'{text}' -> {tokens}")
    print([enc.decode([t]) for t in tokens])