"""
Config file
"""
from pathlib import Path

config_dict = {
    # Load data
    'data_path': Path("data"),
    'url': "http://deeplearning.net/data/mnist/",
    'filename': "mnist.pkl.gz",
    'lr': 0.1,
    'epochs': 10,
    'batch_size': 32
}