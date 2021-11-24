import argparse
import json
import logging
import os
from typing import Tuple

import coloredlogs
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from src.abalone_game import AbaloneGame as Game
from src.coach import CoachArguments
from src.coach import ParallelCoach as Coach
# from src.coach import Coach
from src.neural_net import NNetWrapper as nn

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


def main(coach_arguments: dict = None):
    if coach_arguments:
        args = CoachArguments(**coach_arguments)
    else:
        args = CoachArguments()

    if args.tpu_name:
        log.info('Connecting to TPU %s', args.tpu_name)
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=args.tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g, args)

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
        c.load_train_examples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    # use spawn method to be portable and allow for resource monitoring
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description='Initialize training process')
    parser.add_argument('--args', dest='coach_arguments_path', type=str, default='',
                        help='an integer for the accumulator')
    args = parser.parse_args()

    coach_arguments = {}
    if args.coach_arguments_path:
        log.info('Loading settings from %s', args.coach_arguments_path)
        with file_io.FileIO(args.coach_arguments_path, 'r') as settings_json:
            coach_arguments = json.loads(settings_json.read())
    main(coach_arguments)
