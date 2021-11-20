import os

from abalone_engine.players import RandomPlayer
from src.arena import ParallelArena
from src.coach import CoachArguments


def test_player_games():
    args = CoachArguments(num_arena_workers=2,
                          num_arena_comparisons=2, num_MCTS_sims=2)
    test_net_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'lib/nnet.tar')
    arena = ParallelArena(RandomPlayer, (), {},
                          RandomPlayer, (), {}, args)
    pwins, nwins, draws = arena.player_games()
