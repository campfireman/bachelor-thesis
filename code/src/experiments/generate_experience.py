import logging
import multiprocessing as mp
import time
from collections import deque
from pickle import Pickler

import numpy as np
from abalone_engine.enums import Player
from abalone_engine.game import Game, Move
from abalone_engine.players import AbaProPlayer, AlphaBetaPlayer, RandomPlayer
from src.abalone_game import AbaloneGame
from src.utils import move_standard_to_index

log = logging.getLogger(__name__)


def run_self_play_worker(proc_id: int, train_example_queue: mp.Queue, black_class, black_args, black_kwargs, white_class, white_args, white_kwargs):
    print(f'Worker {proc_id} checking in')

    black = black_class(Player.BLACK, *black_args, **black_kwargs)
    white = white_class(Player.WHITE, *white_args, **white_kwargs)
    abalone_game = AbaloneGame()
    action_size = abalone_game.get_action_size()

    # to be resetted
    game = Game()
    train_examples = []
    moves_history = []
    canonicalBoard = game.canonical_board()
    cur_player = game.turn.value
    episode_step = 0
    start = time.time()

    while True:
        episode_step += 1

        move = black.turn(game, moves_history) if game.turn is Player.BLACK else white.turn(
            game, moves_history)
        moves_history.append(move)
        move_index = move_standard_to_index(
            Move.from_original(move).to_standard())

        pi = np.zeros(action_size)
        pi[move_index] = 1

        sym = abalone_game.get_symmetries(canonicalBoard, pi)
        for b, p in sym:
            train_examples.append([b, cur_player, p, None])

        game.move(*move)
        game.switch_player()
        cur_player = game.turn.value
        canonicalBoard = game.canonical_board()

        r = abalone_game.get_game_ended_limited(
            canonicalBoard, cur_player, episode_step)
        if r != 0:
            end = time.time()
            print(
                f'Finished game with in {(end-start):.2f}s')
            train_example_queue.put(
                [(x[0], x[2], r * ((-1) ** (x[1] != cur_player))) for x in train_examples])

            game = Game()
            canonicalBoard = game.canonical_board()
            cur_player = 1
            episode_step = 0
            moves_history = []
            # switch players
            black_class, black_args, black_kwargs, white_class, white_args, white_kwargs = white_class, white_args, white_kwargs, black_class, black_args, black_kwargs
            black = black_class(Player.BLACK, *black_args, **black_kwargs)
            white = white_class(Player.WHITE, *white_args, **white_kwargs)
            start = time.time()


def spawnum_self_play_workers(no_workers: int, manager: mp.Manager):
    print(
        f'Spawning {no_workers} game workers')

    train_examples_queue = mp.Queue()
    processes = []

    for i in range(0, no_workers):
        print(f'Spawning worker {i}')
        process = mp.Process(
            target=run_self_play_worker,
            args=(
                i, train_examples_queue,
                AbaProPlayer, (), {'depth': 3, 'is_verbose': False},
                RandomPlayer, (), {},
            )
        )
        process.start()
        processes.append(processes)

    return train_examples_queue, processes


def save_buffer(buffer, buffer_filename):
    print(f'Saving buffer to {buffer_filename}')
    with open(buffer_filename, 'wb') as file:
        Pickler(file).dump(buffer)


def main():
    buffer_filename = 'data/heuristic_experience1.buffer'
    buffer = deque()
    num_games = 50
    save_interval = 10
    no_workers = 5

    with mp.Manager() as manager:
        train_example_queue, procs = spawnum_self_play_workers(
            no_workers, manager)

        for i in range(1, num_games + 1):
            buffer.extend(train_example_queue.get())
            print(f'Starting Reading game #{i} ...')
            if i % save_interval == 0:
                save_buffer(buffer, buffer_filename)
        for proc in procs:
            proc.kill()
    save_buffer(buffer, buffer_filename)
