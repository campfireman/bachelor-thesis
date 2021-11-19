import logging
import os
from dataclasses import dataclass
from typing import Tuple

import coloredlogs

from src.abalone_game import AbaloneGame as Game
from src.coach import ParallelCoach as Coach
# from src.coach import Coach
from src.neural_net import NNetWrapper as nn

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


@dataclass
class TrainingArguments:
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


args = TrainingArguments()


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', *args.load_folder_file)
        nnet.load_checkpoint(
            args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main()
