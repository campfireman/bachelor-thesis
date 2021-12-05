import os
import sys
from pickle import Pickler, Unpickler

from src.abalone_game import AbaloneGame


def load_train_examples(path):
    train_examples_history = []
    if not os.path.isfile(path):
        print(f'File "{path}" with trainExamples not found!')
        r = input("Continue? [y|n]")
        if r != "y":
            sys.exit()
    else:
        print("File with trainExamples found. Loading it...")
        with open(path, "rb") as f:
            train_examples_history = Unpickler(f).load()
        print('Loading done!')
    return train_examples_history


def main():
    checkpoints = [
        '../data/2021-11-30-2021-12-01_training/checkpoint_10.pth.tar.examples',
        '../data/2021-11-30-2021-12-01_training/checkpoint_15.pth.tar.examples',
        '../data/2021-11-30-2021-12-01_training/checkpoint_20.pth.tar.examples',
        '../data/2021-11-30-2021-12-01_training/checkpoint_25.pth.tar.examples',
        '../data/2021-11-30-2021-12-01_training/checkpoint_30.pth.tar.examples',
        '../data/2021-11-30-2021-12-01_training/checkpoint_34.pth.tar.examples',
        '../data/2021-12-02_training/checkpoint_10.pth.tar.examples',
        '../data/2021-12-02_training/checkpoint_15.pth.tar.examples',
        '../data/2021-12-02_training/checkpoint_20.pth.tar.examples',
        '../data/2021-12-02_training/checkpoint_25.pth.tar.examples',
        '../data/2021-12-02_training/checkpoint_35.pth.tar.examples',
        'data/temp/checkpoint_1.pth.tar.examples',
    ]
    buffer_path = 'data/filtered_experience.buffer'
    buffer = {}
    g = AbaloneGame()
    for checkpoint in checkpoints:
        train_examples = load_train_examples(checkpoint)
        for game in train_examples:
            for experience in game:
                if 0.01 > abs(experience[2]):
                    continue
                s = g.string_representation(experience[0])
                if s in buffer:
                    if abs(buffer[s][2]) > abs(experience[2]):
                        continue
                buffer[s] = experience
    with open(buffer_path, "wb") as f:
        Pickler(f).dump(list(buffer.values()))
