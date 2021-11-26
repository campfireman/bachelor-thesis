import os

from abalone_engine.players import AbaProPlayer, AlphaBetaPlayer, RandomPlayer
from src.abalone_game import AbaloneNNPlayer
from src.arena import ParallelArena
from src.settings import CoachArguments


def test_play_games():
    test_net_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'lib/nnet.tar')
    arena = ParallelArena(
        AbaloneNNPlayer,
        (),
        {'nnet_fullpath': os.path.join(os.path.dirname(
            __file__), 'lib/temp.pth.tar'), 'args': CoachArguments(num_MCTS_sims=4)},
        RandomPlayer,
        (),
        {},
        1,
        1,
        verbose=True
    )
    pwins, nwins, draws, _, _ = arena.play_games()
