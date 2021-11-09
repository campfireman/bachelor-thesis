import sys

sys.path.append('..')
sys.path.append('../..')

from alpha_zero_general.utils import dotdict
from src.Abalone import AbaloneGame
from src.AbaloneNN import AbaloneNN

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
