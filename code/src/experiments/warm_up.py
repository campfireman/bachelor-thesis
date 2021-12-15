import logging
import os
import time
from dataclasses import dataclass
from pickle import Unpickler
from random import shuffle

import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from abalone_engine.players import RandomPlayer
from src.abalone_game import AbaloneGame, AbaloneNNPlayer
from src.arena import ParallelArena as Arena
from src.experiments.generate_experience import generate_experience
from src.neural_net_torch import NNetWrapper
# from src.neural_net import NNetWrapper
from src.settings import CoachArguments
from src.utils import CsvTable


@dataclass
class WarmUpArgs:
    old_name: str
    buffer_path: str
    load_buffer_from_file: bool
    iterations: int = 5
    train: bool = True
    load_old: bool = False


def load_buffer(path):
    buffer = []
    with open(path, 'rb') as file:
        buffer = Unpickler(file).load()
    return buffer


log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


def main():
    args = WarmUpArgs(
        train=False,
        load_old=True,
        load_buffer_from_file=False,
        iterations=25,
        old_name='heuristic_warm_up_net_large.pth.tar',
        buffer_path='data/heuristic_experienceX.buffer',
        # buffer_path = 'data/filtered_experience1.buffer',
    )
    c_args = CoachArguments(
        num_random_agent_comparisons=10,
        num_arena_workers=2,
        nnet_size='mini',
        residual_tower_size=10,
        framework='torch',
        gpus_arena=['0'],
        num_MCTS_sims=30,
        num_channels=768,
    )
    if c_args.framework == 'tensorflow':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    game = AbaloneGame()

    NNET_NAME_CURRENT = 'heuristic_warm_up_net_large.pth.tar'
    random_player_game_stats_csv = CsvTable(
        'data',
        'warm_up_random_player_performance.csv',
        ['iteration', 'timestamp', 'nnet_wins', 'random_wins',
            'draws', 'nnet_cumul_rewards', 'rndm_cumul_rewards'],
    )
    if args.train:
        if args.load_buffer_from_file:
            train_examples = load_buffer(args.buffer_path)
        else:
            train_examples = generate_experience()
        nnet = NNetWrapper(game, c_args)
        nnet.show_info()
        boards, pis, zs = zip(*train_examples)
        plt.hist(zs, density=False)
        plt.title("histogram")
        plt.show()
    if args.train and args.load_old:
        nnet.load_checkpoint(
            folder=c_args.checkpoint, filename=args.old_name)

    for i in range(0, args.iterations):
        if args.train:
            shuffle(train_examples)

            nnet.train(train_examples)
            nnet.save_checkpoint(
                folder=c_args.checkpoint, filename=NNET_NAME_CURRENT)
        else:
            print('Skipping training')
        print('Starting comparision with random player')
        arena = Arena(
            AbaloneNNPlayer,
            (),
            {'nnet_fullpath': os.path.join(c_args.checkpoint, NNET_NAME_CURRENT),
                'args': c_args},
            RandomPlayer,
            (),
            {},
            c_args,
            verbose=False
        )
        nwins, rwins, draws, nrewards, rrewards = arena.play_games(
            c_args.num_random_agent_comparisons)
        random_player_game_stats_csv.add_row(
            [i, time.time(), nwins, rwins, draws, nrewards, rrewards])
        print('NN/RNDM WINS : %d / %d ; DRAWS : %d' %
              (nwins, rwins, draws))
        if args.train and not args.load_buffer_from_file:
            train_examples.extend(generate_experience())


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    main()
