from datasets import load_dataset, load_dataset_builder


def load_dataset() -> str:
    with open("input.txt", "r") as f:
        text = (f.read())

    return text