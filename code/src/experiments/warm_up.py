import logging
import os
import time
from dataclasses import dataclass
from pickle import Unpickler
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from abalone_engine.players import RandomPlayer
from src.abalone_game import AbaloneGame, AbaloneNNPlayer
from src.arena import ParallelArena as Arena
from src.neural_net_torch import NNetWrapper
from src.settings import CoachArguments
from src.utils import CsvTable


@dataclass
class WarmUpArgs:
    iterations: int = 5


def load_buffer(path):
    buffer = []
    with open(path, 'rb') as file:
        buffer = Unpickler(file).load()
    return buffer


log = logging.getLogger(__name__)


def main():
    import multiprocessing as mp
    mp.set_start_method('spawn')
    train_examples = load_buffer('data/filtered_experience.buffer')
    args = WarmUpArgs()
    c_args = CoachArguments(
        num_random_agent_comparisons=10,
        num_arena_workers=2,
        arena_worker_cpu=False,
        nnet_size='small',
        num_MCTS_sims=200,
    )
    game = AbaloneGame()
    nnet = NNetWrapper(game, c_args)
    NNET_NAME_CURRENT = 'warm_up_net.pth.tar'
    random_player_game_stats_csv = CsvTable(
        'data',
        'warm_up_random_player_performance.csv',
        ['iteration', 'timestamp', 'nnet_wins', 'random_wins',
            'draws', 'nnet_cumul_rewards', 'rndm_cumul_rewards'],
    )
    boards, pis, zs = zip(*train_examples)
    # plt.hist(zs, bins=[-1/2, -1/3, -1/6, -1e-08,
    #          1e-08, 1/6, 1/3, 1/2], density=False)
    plt.hist(zs, density=False)
    plt.title("histogram")
    plt.show()

    for i in range(0, args.iterations):
        # shuffle(train_examples)

        # nnet.train(train_examples)
        # nnet.save_checkpoint(
        #     folder=c_args.checkpoint, filename=NNET_NAME_CURRENT)
        print('Starting comparision with random player')
        arena = Arena(
            AbaloneNNPlayer,
            (),
            {'nnet_fullpath': os.path.join(c_args.checkpoint, NNET_NAME_CURRENT),
                'args': c_args},
            RandomPlayer,
            (),
            {},
            c_args.num_random_agent_comparisons,
            c_args.num_arena_workers,
            verbose=False
        )
        nwins, rwins, draws, nrewards, rrewards = arena.play_games()
        random_player_game_stats_csv.add_row(
            [i, time.time(), nwins, rwins, draws, nrewards, rrewards])
        print('NN/RNDM WINS : %d / %d ; DRAWS : %d' %
              (nwins, rwins, draws))
