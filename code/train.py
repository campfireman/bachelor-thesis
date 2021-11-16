import logging
import os

import coloredlogs
from alpha_zero_general.utils import dotdict

from src.abalone_game import AbaloneGame as Game
from src.coach import Coach
from src.neural_net import NNetWrapper as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    # Number of complete self-play games to simulate during a new iteration.
    'numEps': 20,
    'tempThreshold': 15,        #
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'updateThreshold': 0.6,
    # Number of game examples to train the neural networks.
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    # Number of games to play during arena play to determine if new net will be accepted.
    'arenaCompare': 6,
    'cpuct': 1,

    'checkpoint': './tempX/',
    'load_model': False,
    'load_folder_file': ('/home/ture/projects/bachelor-thesis/src/temp', 'temp.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


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
    main()
