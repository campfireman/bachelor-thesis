import os

from abalone_engine.players import AbaProPlayer, RandomPlayer
from src.arena import ParallelArena


def test_player_games():
    test_net_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'lib/nnet.tar')
    arena = ParallelArena(
        RandomPlayer, (), {},
        RandomPlayer, (), {},
        2,
        2,)
    pwins, nwins, draws = arena.player_games()
