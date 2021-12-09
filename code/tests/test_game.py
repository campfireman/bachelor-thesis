import time

import numpy as np
from abalone_engine.game import Game
from numpy.lib.function_base import average
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
    player = 1
    board = TEST_BOARD
    pi = np.random.uniform(0.0, 1.0, size=game.get_action_size())
    valids = game.get_valid_moves(TEST_BOARD, player)
    pi = valids * pi
    start = time.time()
    symmetries = list(game.get_symmetries(TEST_BOARD, pi))
    end = time.time()
    print(f'time: {end- start}')
    r = game.get_game_ended(TEST_BOARD, player)
    lens = []
    count = 0

    while r == 0:
        count += 1
        if count > 200:
            break
        move = np.random.choice(np.array(np.argwhere(valids)).flatten())
        board, player = game.get_next_state(board, player, move)
        symmetries = game.get_symmetries(board, pi)
        lens.append(len(list(symmetries)))
        r = game.get_game_ended(board, player)
        pi = np.random.uniform(0.0, 1.0, size=game.get_action_size())
        valids = game.get_valid_moves(board, player)
        pi = valids * pi
    print(f'Average size of symmetries: {np.average(lens)}')
