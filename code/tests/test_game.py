import time

import numpy as np
from abalone_engine.game import Game
from src.abalone_game import AbaloneGame


def test_get_symmetries():
    TEST_BOARD = np.array([
        # 0  1  2  3  4  5  6  7  8
        [0, 0, 0, 0, -1, -1, -1, -1, -1],  # 0
        [0, 0, 0, -1, -1, -1, -1, -1, -1],  # 1
        [0, 0, 0, 0, -1, -1, -1, 0, 0],  # 2
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
        [0, 0, 1, 1, 1, 0, 0, 0, 0],  # 6
        [1, 1, 1, 1, 1, 1, 0, 0, 0],  # 7
        [1, 1, 1, 1, 1, 0, 0, 0, 0],  # 8
    ], dtype='int')
    game = AbaloneGame()
    pi = np.random.uniform(0.0, 1.0, size=game.get_action_size())
    valids = game.get_valid_moves(TEST_BOARD, 1)
    pi = valids * pi
    start = time.time()
    symmetries = game.get_symmetries(TEST_BOARD, pi)
    end = time.time()
    print(f'time: {end- start}')
    assert len(symmetries) == 6
