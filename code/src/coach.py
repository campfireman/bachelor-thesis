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
from alpha_zero_general.Coach import Coach
from alpha_zero_general.Game import Game
from alpha_zero_general.utils import dotdict
from tqdm import tqdm

from src.neural_net import NNetWrapper

from .arena import Arena
from .mcts import MCTS

log = logging.getLogger(__name__)


@dataclass
class CoachArguments:
    numIters: int = 1000
    numEps: int = 10
    tempThreshold: int = 15
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    updateThreshold: float = 0.6
    # Number of game examples to train the neural networks.
    maxlenOfQueue: int = 2000000
    # Number of games moves for MCTS to simulate.
    numMCTSSims: int = 2
    # Number of games to play during arena play to determine if new net will be accepted.
    arenaCompare: int = 6
    cpuct: float = 1
    n_self_play_workers: int = 4

    checkpoint: str = './temp/'
    load_model: bool = False
    load_folder_file: Tuple[str, str] = (
        '/home/ture/projects/bachelor-thesis/code/src/temp', 'temp.pth.tar')
    numItersForTrainExamplesHistory: int = 20


class ParallelCoach(Coach):
    NNET_NAME_CURRENT = 'temp.pth.tar'
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
        board = game.getInitBoard()
        cur_player = 1
        episode_step = 0
        start = time.time()

        while True:
            mcts = MCTS(game, nnet, args)
            episode_step += 1
            canonicalBoard = game.getCanonicalForm(board, cur_player)
            temp = int(episode_step < args.tempThreshold)

            pi = mcts.getActionProb(canonicalBoard, temp=temp)
            sym = game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                train_examples.append([b, cur_player, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, cur_player = game.getNextState(
                board, cur_player, action)

            r = game.getGameEndedLimited(
                board, cur_player, episode_step)
            if r != 0:
                end = time.time()
                log.info(
                    f'Finished game with nnet id: {nnet_id.value} in {(end-start):.2f}s')
                train_example_queue.put(
                    [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in train_examples])
                board = game.getInitBoard()
                cur_player = 1
                episode_step = 0
                if nnet_id.value > cur_nnet_id:
                    log.info(
                        f'[{proc_id}] Loading Neural Net with ID: {nnet_id.value}')
                    cur_nnet_id = update_nnet(nnet)
                start = time.time()

    def spawn_self_play_workers(self, no_workers: int, manager: mp.Manager) -> Tuple[mp.Queue, mp.Value]:
        log.info(f'Spawning {self.args.n_self_play_workers} self play workers')

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
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        self.initialize_nnet()
        with mp.Manager() as manager:
            train_example_queue, nnet_id = self.spawn_self_play_workers(
                self.args.n_self_play_workers, manager)

            # wait for first round of games to finish
            while train_example_queue.qsize() < self.args.numEps:
                log.info(f'Not enough train examples waiting')
                time.sleep(10.0)

            for i in range(1, self.args.numIters + 1):
                log.info(f'Starting Iter #{i} ...')

                trainExamples = []
                iteration_examples = deque([], self.args.maxlenOfQueue)
                while not train_example_queue.empty():
                    game = train_example_queue.get()
                    trainExamples.extend(game)
                    iteration_examples.append(game)
                shuffle(trainExamples)
                # save the iteration examples to the history
                self.trainExamplesHistory.append(iteration_examples)

                if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                    log.warning(
                        f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                    self.trainExamplesHistory.pop(0)
                # backup history to a file
                # NB! the examples were collected using the model from the previous iteration, so (i-1)
                self.saveTrainExamples(i - 1)

                # training new network, keeping a copy of the old one
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint, filename=self.NNET_NAME_CURRENT)
                self.pnet.load_checkpoint(
                    folder=self.args.checkpoint, filename=self.NNET_NAME_CURRENT)
                pmcts = MCTS(self.game, self.pnet, self.args)

                self.nnet.train(trainExamples)
                nmcts = MCTS(self.game, self.nnet, self.args)

                log.info('PITTING AGAINST PREVIOUS VERSION')
                arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                              lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game, display=self.game.display_state)
                pwins, nwins, draws = arena.playGames(
                    self.args.arenaCompare, verbose=False)

                log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' %
                         (nwins, pwins, draws))
                if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
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
