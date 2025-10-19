
class simpleTokenizer():
    def __init__(self, text: str) -> None:
        self.vocab = sorted(list(set(text)))    

        self.ctoi = {c: idx for idx, c in enumerate(self.vocab)}
        self.itoc = {idx: c for idx, c in enumerate(self.vocab)}

        self.encode = lambda string: [self.ctoi[char] for char in string]
        self.decode = lambda ints: "".join([self.itoc[i] for i in ints])
        
    def encode(self, text: str) -> list[int]:
        return self.encode(text)
    
    def decode(self, sequence: list[int]) -> str:
        return self.decode(sequence)