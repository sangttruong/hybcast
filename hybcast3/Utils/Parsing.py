from argparse import ArgumentParser, Namespace
import torch

def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.
    :param parser: An ArgumentParser.
    """
    # General arguments

    parser.add_argument('--data_path', type=str, help='Path to data CSV file')


def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).
    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()

    return args