"""
Main file to execute NN models from PyTorch
Author: Jonathan Tremblay
Core code taken from: https://pytorch.org/tutorials/beginner/nn_tutorial.html

To execute the file in terminal:
python main.py --configfile /home/john/Desktop/PyCharmProjects/ExploringPyTorch/basic_nn/configurations/config_linear_model.py

"""

import argparse
import os
from utils.config_reader import read_config
from utils.data_loader import DataManipulation
from models.linear_model import LinearModel

if __name__ == '__main__':
    # Fetch configuration file and execute variables
    parser = argparse.ArgumentParser(description='Configurations to run a model')
    parser.add_argument('-c', '--configfile', default=f"{os.getcwd()}/configurations/config_linear_model.py",
                        help='file to read the config from')
    args = vars(parser.parse_args())
    read = read_config(args['configfile'])
    locals().update(read)

    # Data loading
    data = DataManipulation(name='mnist')
    data.loader_mnist(data_path=data_path, filename=filename, url=url)
    x_train, y_train, x_valid, y_valid = data.split_mnist(data_path=data_path, filename=filename)

    # Creating and training model
    lin_model = LinearModel(size1=x_train.shape[1], size2=10, lr=lr)
    lin_model.training(train_input=x_train, train_targets=y_train, eval_input=x_valid, eval_targets=y_valid,
                       epochs=epochs, batch_size=batch_size, verbose=False)
    print(lin_model.evaluate(input=x_valid, targets=y_valid))




