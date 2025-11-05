from src.data.tokenizer import SimpleTokenizer
from torch import stack, randint


class GenericDataset:
    def __init__(self, split_ratio:float = 0.9, batch_size: int = 4, ctx_len: int = 8) -> None:
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.ctx_len = ctx_len
        
        self.raw_text = ""
        self.data = []

        self.tokenizer = None

        self.train_data = None
        self.test_data = None
        
    def _prepare_data(self) -> None:
        pass

    def get_batch(self, split="train") -> None:
        pass


class ShakespeareDataset(GenericDataset):
    def __init__(self) -> None:
        super().__init__()
        self._prepare_data()
        
    def _prepare_data(self, path="./src/data/shakespeare.txt"):
        with open(path, "r") as f:
            self.raw_text = f.read()
            self.tokenizer = SimpleTokenizer(self.raw_text) 
            data = self.tokenizer.encode(self.raw_text)

            n = int(self.split_ratio * len(self.raw_text))
            self.train_data = data[:n]
            self.test_data = data[n:]

    def get_batch(self, split="train"):
        data = self.train_data if split=="train" else self.test_data
        idx = randint(len(data) - self.ctx_len, (self.batch_size,), )
        x = stack([data[i: i + self.ctx_len] for i in idx])
        y = stack([data[i + 1: i + self.ctx_len + 1] for i in idx])
        
        return x, y


if __name__ == "__main__":
    xb,yb = ShakespeareDataset().get_batch()
    print(xb.shape)
    print(yb.shape)

    # for i in range(4):
    #     for l in range (8):
    #         print(f"When input is {xb[i, :l+1].tolist()}, target is {yb[i, l]}")

    from src.model.gpt import VanillaGPT

    out = VanillaGPT().forward(xb)

    print(out.shape)