import logging
import multiprocessing as mp
import os

import numpy as np
from tqdm import tqdm

from src.mcts import MCTS
from src.neural_net import NNetWrapper

log = logging.getLogger(__name__)


class ParallelArena():
    def __init__(self, player1, player2, game, args, display=None, workers=1, verbose=False):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.args = args
        self.display = display
        self.workers = workers
        self.verbose = verbose

    def play_game(self, n: int, cpu: bool = True):
        """
        Executes one episode of a game.
        draw result returned from the game that is neither 1, -1, nor 0.
        """
        if cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        nnet = NNetWrapper(self.game)
        nnet.load_checkpoint(full_path=self.player1)
        nmcts = MCTS(self.game, nnet, self.args)
        pnet = NNetWrapper(self.game)
        pmcts = MCTS(self.game, pnet, self.args)
        nnet.load_checkpoint(full_path=self.player2)
        players = [lambda x: np.argmax(pmcts.get_action_prob(
            x, temp=0)), None, lambda x: np.argmax(nmcts.get_action_prob(x, temp=0))]
        cur_player = 1
        board = self.game.get_init_board()
        it = 0
        while self.game.get_game_ended_limited(board, cur_player, it) == 0:
            it += 1
            action = players[cur_player +
                             1](self.game.get_canonical_form(board, cur_player))

            valids = self.game.get_valid_moves(
                self.game.get_canonical_form(board, cur_player), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board, cur_player = self.game.get_next_state(
                board, cur_player, action)
        if self.verbose:
            assert self.display
            log.info(
                f"Game over: Turn {str(it)} Result: {str(self.game.get_game_ended_limited(board, 1, it))}")
            self.display(board)
        return cur_player * self.game.get_game_ended_limited(board, cur_player, it)

    def player_games(self, num: int, verbose: bool = False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        num = int(num / 2)
        one_won = 0
        two_won = 0
        draws = 0
        tolerance = 0.01

        with mp.Pool(processes=self.workers) as pool:
            results = pool.map(self.play_game, range(0, num))

            for game_result in tqdm(results, desc="Arena.player_games (1)"):
                if game_result > tolerance:
                    one_won += 1
                elif game_result < -tolerance:
                    two_won += 1
                else:
                    draws += 1
            results = pool.map(self.play_game, range(0, num))
            pool.map(self.play_game, results)

            self.player1, self.player2 = self.player2, self.player1

            for game_result in tqdm(results, desc="Arena.player_games (2)"):
                if game_result > tolerance:
                    two_won += 1
                elif game_result < -tolerance:
                    one_won += 1
                else:
                    draws += 1

            return one_won, two_won, draws
