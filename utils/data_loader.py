import requests
import pickle
import gzip
import torch


class DataManipulation:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'{__class__.__name__}({self.name})'

    @staticmethod
    def loader_mnist(data_path, filename, url):
        path = data_path / "mnist"
        path.mkdir(parents=True, exist_ok=True)

        if not (path / filename).exists():
            content = requests.get(url + filename).content
            (path / filename).open("wb").write(content)

    @staticmethod
    def split_mnist(data_path, filename):
        path = data_path / "mnist"
        with gzip.open((path / filename).as_posix(), "rb") as f:
            ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

        x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
        return x_train, y_train, x_valid, y_valid




