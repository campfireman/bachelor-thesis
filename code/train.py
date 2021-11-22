import json
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


def main():
    training_settings = os.environ.get('TRAINING_SETTINGS', None)
    if training_settings:
        log.info('Loading settings from %s', training_settings)
        with open(training_settings, 'r') as settings_json:
            settings = json.loads(settings_json.read())
            args = CoachArguments(**settings)
    else:
        args = CoachArguments()

    if args.tpu_name:
        log.info('Connecting to TPU %s', args.tpu_name)
        import tensorflow as tf
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=args.tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)

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
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main()
