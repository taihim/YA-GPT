import torch

class simpleTokenizer():
    """Basic tokenize for character level encoding."""
    def __init__(self, text: str) -> None:
        self.vocab = sorted(list(set(text)))    

        self.ctoi = {c: idx for idx, c in enumerate(self.vocab)}
        self.itoc = {idx: c for idx, c in enumerate(self.vocab)}

        self.encode = lambda string: torch.tensor([self.ctoi[char] for char in string])
        self.decode = lambda ints: "".join([self.itoc[i] for i in ints])
        