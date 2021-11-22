import logging
import multiprocessing as mp
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from random import shuffle
from typing import Tuple

import numpy as np
from abalone_engine.players import (AbaProPlayer, AlphaBetaPlayerFast,
                                    RandomPlayer)
from alpha_zero_general.Coach import Coach
from alpha_zero_general.Game import Game

from src.abalone_game import AbaloneNNPlayer
from src.neural_net import NNetWrapper
from src.utils import CsvTable

from .arena import ParallelArena as Arena
from .mcts import MCTS

log = logging.getLogger(__name__)


@dataclass
class CoachArguments:
    num_iters: int = 1000
    num_eps: int = 2
    temp_treshhold: int = 15
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    update_treshold: float = 0.6
    # Max number of game examples in the queue between coach and self play workers
    maxlen_of_queue: int = 2000
    # Number of games moves for MCTS to simulate.
    num_MCTS_sims: int = 4
    cpuct: float = 1
    num_self_play_workers: int = 4
    # Number of games to play during arena play to determine if new net will be accepted.
    num_self_comparisons: int = 2
    # At which interval playoffs against heuristic agent and random agent are being performed
    agent_comparisons_step_size: int = 5
    num_random_agent_comparisons: int = 2
    num_heuristic_agent_comparisons: int = 2
    num_arena_workers: int = 2

    data_directory: str = './'
    checkpoint: str = './temp/'
    load_model: bool = False
    load_folder_file: Tuple[str, str] = (
        '/home/ture/projects/bachelor-thesis/code/src/temp', 'temp.pth.tar')
    num_iters_for_train_examples_history: int = 20


class ParallelCoach(Coach):
    NNET_NAME_CURRENT = 'temp.pth.tar'
    NNET_NAME_NEW = 'temp_new.pth.tar'
    NNET_NAME_BEST = 'best.pth.tar'

    @staticmethod
    def run_self_play_worker(proc_id: int, args: CoachArguments, game: Game, train_example_queue: mp.Queue, nnet_path: str, nnet_id: mp.Value, cpu: bool = True):
        def update_nnet(nnet: NNetWrapper) -> int:
            nnet.load_checkpoint(full_path=nnet_path)
            return nnet_id.value

        log.info(f'Worker {proc_id} checking in')
        if cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        nnet = NNetWrapper(game)

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
                log.info(
                    f'Finished game with nnet id: {cur_nnet_id} in {(end-start):.2f}s')
                train_example_queue.put(
                    [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in train_examples])
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
                    i, self.args, self.game,
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
            ['iteration', 'timestamp', 'wins', 'losses', 'draws'],
        )
        heuristic_player_game_stats_csv = CsvTable(
            self.args.data_directory,
            f'{training_start}_heuristic_player_game_stats.csv',
            ['iteration', 'timestamp', 'wins', 'losses', 'draws'],
        )

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
                self.trainExamplesHistory.append(iteration_examples)

                if len(self.trainExamplesHistory) > self.args.num_iters_for_train_examples_history:
                    log.warning(
                        f"Removing the oldest entry in train_examples. len(train_examples_history) = {len(self.trainExamplesHistory)}")
                    self.trainExamplesHistory.pop(0)
                # backup history to a file
                # NB! the examples were collected using the model from the previous iteration, so (i-1)
                self.saveTrainExamples(i - 1)

                train_examples = []
                for e in self.trainExamplesHistory:
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
                     'num_MCTS_sims': self.args.num_MCTS_sims, 'cpuct': self.args.cpuct},
                    AbaloneNNPlayer,
                    (),
                    {'nnet_fullpath': os.path.join(self.args.checkpoint, self.NNET_NAME_NEW),
                     'num_MCTS_sims': self.args.num_MCTS_sims, 'cpuct': self.args.cpuct},
                    self.args.num_self_comparisons,
                    self.args.num_arena_workers,
                    verbose=False
                )
                pwins, nwins, draws = arena.play_games()

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
                        folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
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
                         'num_MCTS_sims': self.args.num_MCTS_sims, 'cpuct': self.args.cpuct},
                        RandomPlayer,
                        (),
                        {},
                        self.args.num_random_agent_comparisons,
                        self.args.num_arena_workers,
                        verbose=False
                    )
                    nwins, rwins, draws = arena.play_games()
                    random_player_game_stats_csv.add_row(
                        [i, time.time(), nwins, rwins, draws])
                    log.info('NN/RNDM WINS : %d / %d ; DRAWS : %d' %
                             (nwins, rwins, draws))

                    log.info('PITTING AGAINST HEURISTIC PLAYER')
                    arena = Arena(
                        AbaloneNNPlayer,
                        (),
                        {'nnet_fullpath': os.path.join(self.args.checkpoint, self.NNET_NAME_CURRENT),
                         'num_MCTS_sims': self.args.num_MCTS_sims, 'cpuct': self.args.cpuct},
                        RandomPlayer,
                        (),
                        {},
                        self.args.num_random_agent_comparisons,
                        self.args.num_arena_workers,
                        verbose=False
                    )
                    nwins, hwins, draws = arena.play_games()
                    heuristic_player_game_stats_csv.add_row(
                        [i, time.time(), nwins, hwins, draws])
                    log.info('NN/HRSTC WINS : %d / %d ; DRAWS : %d' %
                             (nwins, hwins, draws))
                iteration_duration = time.time() - iteration_start
                performance_stats_csv.add_row(
                    [i, time.time(), iteration_duration, training_duration, examples_read_from_queue, len(train_examples)])
