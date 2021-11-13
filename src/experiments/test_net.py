import sys

sys.path.append('..')
sys.path.append('../..')

from alpha_zero_general.utils import dotdict
from src.abalone import AbaloneGame
from src.neural_net import AbaloneNN

nn = AbaloneNN(AbaloneGame(), dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
}))

nn.show_info()
nn.visualize()
