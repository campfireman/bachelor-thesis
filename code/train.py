import logging
import os
from typing import Tuple

import coloredlogs

from src.abalone_game import AbaloneGame as Game
from src.coach import CoachArguments
from src.coach import ParallelCoach as Coach
# from src.coach import Coach
from src.neural_net import NNetWrapper as nn

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


args = CoachArguments()


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
