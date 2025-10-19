from src.tokenizer import simpleTokenizer
from torch import stack, randint
import torch

class shakespeareDataset:
    def __init__(self, path="./src/data/shakespeare.txt", split_ratio: float = 0.9):
        with open(path, "r") as f:
            self.raw_text = f.read()
            self.tokenizer = simpleTokenizer(self.raw_text) 
            data = self.tokenizer.encode(self.raw_text)

            n = int(split_ratio * len(self.raw_text))
            self.train_data = data[:n]
            self.test_data = data[n:]

    def get_batch(self, split="train", ctx_len: int = 8, batch_size:int = 4):
        data = self.train_data if split=="train" else self.test_data
        idx = randint(len(data) - ctx_len, (batch_size,), )
        x = stack([data[i: i + ctx_len] for i in idx])
        y = stack([data[i + 1: i + ctx_len + 1] for i in idx])
        
        return x, y


if __name__ == "__main__":
    xb,yb = shakespeareDataset().get_batch()
    print(xb.shape)
    print(yb.shape)

    for i in range(4):
        for l in range (8):
            print(f"When input is {xb[i, :l+1].tolist()}, target is {yb[i, l]}")