import logging
import multiprocessing as mp
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pickle import Pickler, Unpickler
from random import shuffle
from typing import Tuple

import numpy as np
# import torch.multiprocessing as mp
from abalone_engine.players import AbaProPlayer, AlphaBetaPlayer, RandomPlayer
from alpha_zero_general.Coach import Coach
from alpha_zero_general.Game import Game
from tensorflow.python.lib.io import file_io

from src.abalone_game import AbaloneNNPlayer
from src.mcts import MCTS
from src.neural_net import NNetWrapperBase
from src.settings import CoachArguments
from src.utils import CsvTable

from .arena import ParallelArena as Arena

log = logging.getLogger(__name__)


class ParallelCoach:
    NNET_NAME_CURRENT = 'temp.pth.tar'
    NNET_NAME_NEW = 'temp_new.pth.tar'
    NNET_NAME_BEST = 'best.pth.tar'

    def __init__(self, game: Game, nnet: NNetWrapperBase, args: CoachArguments):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.pnet = self.nnet.__class__(
            self.game, self.args)  # the competitor network
        self.mcts = MCTS(self.game, self.nnet, self.args)
        # history of examples from args.num_iters_for_train_examples latest iterations
        self.train_examples_history = []
        self.skip_first_self_play = False  # can be overriden in load_train_examples()

    def get_checkpoint_file(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def save_train_examples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(
            folder, self.get_checkpoint_file(iteration) + ".examples")
        with file_io.FileIO(filename, "wb+") as f:
            Pickler(f).dump(self.train_examples_history)

    def load_train_examples(self):
        modelFile = os.path.join(
            self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with file_io.FileIO(examplesFile, "rb") as f:
                self.train_examples_history = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skip_first_self_play = True

    @staticmethod
    def run_self_play_worker(proc_id: int, args: CoachArguments, game: Game, nnet_class: NNetWrapperBase, train_example_queue: mp.Queue, nnet_path: str, nnet_id: mp.Value):
        def update_nnet(nnet: NNetWrapperBase) -> int:
            nnet.load_checkpoint(full_path=nnet_path)
            return nnet_id.value

        log.info(f'Worker {proc_id} checking in')
        if args.self_play_worker_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            args.cuda = False
        nnet = nnet_class(game, args)

        cur_nnet_id = update_nnet(nnet)
        train_examples = []
        board = game.get_init_board()
        cur_player = 1
        episode_step = 0
        start = time.time()

        while True:
            mcts = MCTS(game, nnet, args)
            episode_step += 1
            canonicalBoard = game.get_canonical_form(board, cur_player)
            temp = int(episode_step < args.temp_treshhold)

            pi = mcts.get_action_prob(canonicalBoard, temp=temp)
            sym = game.get_symmetries(canonicalBoard, pi)
            for b, p in sym:
                train_examples.append([b, cur_player, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, cur_player = game.get_next_state(
                board, cur_player, action)

            r = game.get_game_ended_limited(
                board, cur_player, episode_step)
            if r != 0:
                end = time.time()
                if not args.filter_by_reward_threshold or (args.filter_by_reward_threshold and abs(r) > 0.001):
                    log.info(
                        f'Finished game with nnet id: {cur_nnet_id} in {(end-start):.2f}s')
                    train_example_queue.put(
                        [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in train_examples])
                else:
                    log.info(
                        f'Discarding game with r = {r}: nnet: {cur_nnet_id} in {(end-start):.2f}s')
                board = game.get_init_board()
                cur_player = 1
                episode_step = 0
                if nnet_id.value > cur_nnet_id:
                    log.info(
                        f'[{proc_id}] Loading Neural Net with ID: {nnet_id.value}')
                    cur_nnet_id = update_nnet(nnet)
                start = time.time()

    def spawnum_self_play_workers(self, no_workers: int, manager: mp.Manager) -> Tuple[mp.Queue, mp.Value]:
        log.info(
            f'Spawning {self.args.num_self_play_workers} self play workers')

        train_examples_queue = mp.Queue()
        nnet_id = manager.Value(value=0, typecode='int')
        nnet_path = os.path.join(self.args.checkpoint, self.NNET_NAME_BEST)

        for i in range(0, no_workers):
            log.info(f'Spawning worker {i}')
            process = mp.Process(
                target=ParallelCoach.run_self_play_worker,
                args=(
                    i, self.args, self.game, self.nnet.__class__,
                    train_examples_queue, nnet_path, nnet_id
                )
            )
            process.start()

        return train_examples_queue, nnet_id

    def initialize_nnet(self):
        # save neural net so it can be loaded by workers
        self.nnet.save_checkpoint(
            folder=self.args.checkpoint, filename=self.NNET_NAME_BEST)

    def learn(self):
        """
        Performs num_iters iterations with num_eps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in train_examples (which has a maximum length of maxlen_of_queue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= update_treshold fraction of games.
        """
        self.initialize_nnet()

        # collect stats in csv files
        training_start = time.time()
        performance_stats_csv = CsvTable(
            self.args.data_directory,
            f'{training_start}_performance_stats.csv',
            ['iteration', 'timestamp', 'iteration_duration',
                'training_duration', 'examples_read_from_queue', 'length_train_examples'],
        )
        random_player_game_stats_csv = CsvTable(
            self.args.data_directory,
            f'{training_start}_random_player_game_stats.csv',
            ['iteration', 'timestamp', 'wins', 'losses', 'draws',
                'nnet_cumul_rewards', 'random_cumul_rewards'],
        )
        heuristic_player_game_stats_csv = CsvTable(
            self.args.data_directory,
            f'{training_start}_heuristic_player_game_stats.csv',
            ['iteration', 'timestamp', 'wins', 'losses', 'draws',
                'nnet_cumul_rewards', 'random_cumul_rewards'],
        )
        self.args.save(training_start)

        with mp.Manager() as manager:
            train_example_queue, nnet_id = self.spawnum_self_play_workers(
                self.args.num_self_play_workers, manager)

            # wait for first round of games to finish
            while train_example_queue.qsize() < self.args.num_eps:
                log.info(f'Not enough train examples waiting')
                time.sleep(10.0)

            for i in range(1, self.args.num_iters + 1):
                iteration_start = time.time()
                log.info(f'Starting Iter #{i} ...')

                iteration_examples = deque([], self.args.maxlen_of_queue)

                examples_read_from_queue = 0
                while not train_example_queue.empty():
                    game = train_example_queue.get()
                    iteration_examples += game
                    examples_read_from_queue += 1
                log.info(
                    f'Loaded {examples_read_from_queue} self play games from queue')

                # save the iteration examples to the history
                self.train_examples_history.append(iteration_examples)

                if len(self.train_examples_history) > self.args.num_iters_for_train_examples_history:
                    log.warning(
                        f"Removing the oldest entry in train_examples. len(train_examples_history) = {len(self.train_examples_history)}")
                    self.train_examples_history.pop(0)
                # backup history to a file
                # NB! the examples were collected using the model from the previous iteration, so (i-1)
                self.save_train_examples(i - 1)

                train_examples = []
                for e in self.train_examples_history:
                    train_examples.extend(e)
                shuffle(train_examples)

                # training new network, keeping a copy of the old one
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename=self.NNET_NAME_CURRENT)
                self.pnet.load_checkpoint(
                    folder=self.args.checkpoint, filename=self.NNET_NAME_CURRENT)

                training_duration_start = time.time()
                self.nnet.train(train_examples)
                training_duration = time.time() - training_duration_start
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename=self.NNET_NAME_NEW)

                log.info('PITTING AGAINST PREVIOUS VERSION')
                arena = Arena(
                    AbaloneNNPlayer,
                    (),
                    {'nnet_fullpath': os.path.join(self.args.checkpoint, self.NNET_NAME_CURRENT),
                     'args': self.args},
                    AbaloneNNPlayer,
                    (),
                    {'nnet_fullpath': os.path.join(self.args.checkpoint, self.NNET_NAME_NEW),
                     'args': self.args},
                    self.args.num_self_comparisons,
                    self.args.num_arena_workers,
                    verbose=False
                )
                pwins, nwins, draws, _, _ = arena.play_games()

                log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' %
                         (nwins, pwins, draws))
                if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.update_treshold:
                    log.info('REJECTING NEW MODEL')
                    self.nnet.load_checkpoint(
                        folder=self.args.checkpoint, filename=self.NNET_NAME_CURRENT)
                else:
                    log.info('ACCEPTING NEW MODEL')
                    nnet_id.value += 1
                    self.nnet.save_checkpoint(
                        folder=self.args.checkpoint, filename=self.get_checkpoint_file(i))
                    self.nnet.save_checkpoint(
                        folder=self.args.checkpoint, filename=self.NNET_NAME_BEST)

                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename=self.NNET_NAME_CURRENT)

                if i == 1 or i % self.args.agent_comparisons_step_size == 0:
                    log.info('PITTING AGAINST RANDOM PLAYER')
                    arena = Arena(
                        AbaloneNNPlayer,
                        (),
                        {'nnet_fullpath': os.path.join(self.args.checkpoint, self.NNET_NAME_CURRENT),
                         'args': self.args},
                        RandomPlayer,
                        (),
                        {},
                        self.args.num_random_agent_comparisons,
                        self.args.num_arena_workers,
                        verbose=False
                    )
                    nwins, rwins, draws, nrewards, rrewards = arena.play_games()
                    random_player_game_stats_csv.add_row(
                        [i, time.time(), nwins, rwins, draws, nrewards, rrewards])
                    log.info('NN/RNDM WINS : %d / %d ; DRAWS : %d' %
                             (nwins, rwins, draws))

                    log.info('PITTING AGAINST HEURISTIC PLAYER')
                    arena = Arena(
                        AbaloneNNPlayer,
                        (),
                        {'nnet_fullpath': os.path.join(self.args.checkpoint, self.NNET_NAME_CURRENT),
                         'args': self.args},
                        AlphaBetaPlayer,
                        (),
                        {},
                        self.args.num_random_agent_comparisons,
                        self.args.num_arena_workers,
                        verbose=False
                    )
                    nwins, hwins, draws, nrewards, hrewards = arena.play_games()
                    heuristic_player_game_stats_csv.add_row(
                        [i, time.time(), nwins, hwins, draws, nrewards, hrewards])
                    log.info('NN/HRSTC WINS : %d / %d ; DRAWS : %d' %
                             (nwins, hwins, draws))
                iteration_duration = time.time() - iteration_start
                performance_stats_csv.add_row(
                    [i, time.time(), iteration_duration, training_duration, examples_read_from_queue, len(train_examples)])
