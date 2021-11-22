import os

from abalone_engine.players import AbaProPlayer, AlphaBetaPlayer, RandomPlayer
from src.arena import ParallelArena


def test_play_games():
    test_net_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'lib/nnet.tar')
    arena = ParallelArena(
        AlphaBetaPlayer,
        (),
        {'depth': 2},
        RandomPlayer,
        (),
        {},
        1,
        1,
        verbose=True
    )
    pwins, nwins, draws, _, _ = arena.play_games()
