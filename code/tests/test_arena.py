import os

from abalone_engine.players import AbaProPlayer, AlphaBetaPlayer, RandomPlayer
from src.abalone_game import AbaloneNNPlayer
from src.arena import ParallelArena
from src.settings import CoachArguments


def test_play_games():
    test_net_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'lib/best.pth.tar')
    arena = ParallelArena(
        AbaloneNNPlayer,
        (),
        {'nnet_fullpath': test_net_path,
            'args': CoachArguments(num_MCTS_sims=1, nnet_size='small', framework='torch')},
        RandomPlayer,
        (),
        {},
        1,
        1,
        verbose=True
    )
    pwins, nwins, draws, _, _ = arena.play_games()
