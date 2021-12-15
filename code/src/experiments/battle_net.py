import logging
from itertools import product
from typing import List, Tuple

from abalone_engine.enums import Player
from abalone_engine.game import Game
from abalone_engine.players import (AbaProPlayer, AbstractPlayer,
                                    AlphaBetaPlayer, RandomPlayer)
from abalone_engine.utils import MoveStats
from src.abalone_game import AbaloneNNPlayer
from src.settings import CoachArguments

log = logging.getLogger(__name__)


def main():
    stats = []
    # black = AbaProPlayer(Player.BLACK, is_verbose=False, depth=3)
    # black = RandomPlayer(Player.BLACK)
    # black = AbaloneNNPlayer(Player.WHITE, '/home/ture/projects/bachelor-thesis/code/data/temp/best.pth.tar',
    #                         CoachArguments(framework='torch', nnet_size='mini', cuda=True, num_MCTS_sims=60))
    # black = AbaloneNNPlayer(Player.BLACK, '/home/ture/projects/bachelor-thesis/code/data/temp/heuristic_warm_up_net_large.pth.tar',
    #                         CoachArguments(framework='torch', nnet_size='mini', cuda=True, num_channels=768, num_MCTS_sims=1000))
    black = AbaloneNNPlayer(Player.BLACK, '/home/ture/best.pth.tar',
                            CoachArguments(framework='torch', nnet_size='mini', num_channels=512, cuda=True, num_MCTS_sims=1000))
    # white = AbaloneNNPlayer(Player.WHITE, '/home/ture/projects/bachelor-thesis/code/data/temp/heuristic_warm_up_net.pth.tar',
    #                         CoachArguments(framework='torch', nnet_size='mini', cuda=True, num_MCTS_sims=60))
    # white = RandomPlayer(Player.WHITE)
    # white = AlphaBetaPlayer(Player.BLACK)
    white = AbaProPlayer(Player.WHITE, is_verbose=False, depth=2)
    game, moves_history, move_stats = Game.run_game(
        black, white, is_verbose=True)


if __name__ == '__main__':
    main()
