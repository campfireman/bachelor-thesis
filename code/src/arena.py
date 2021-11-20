import logging
import multiprocessing as mp
import os
from typing import Tuple

import numpy as np
from abalone_engine.enums import Player
from abalone_engine.game import Game

log = logging.getLogger(__name__)


class ParallelArena():
    def __init__(
        self,
        player1_class: object,
        player1_args: Tuple,
        player1_kwargs: dict,
        player2_class: object,
        player2_args: Tuple,
        player2_kwargs: dict,
        args,
        verbose=False
    ):
        self.player1_class = player1_class
        self.player1_args = player1_args
        self.player1_kwargs = player1_kwargs
        self.player2_class = player2_class
        self.player2_args = player2_args
        self.player2_kwargs = player2_kwargs
        self.args = args
        self.verbose = verbose

    def print_game_result(self, game: Game):
        score = game.get_score()
        score_str = f'BLACK {score[0]} - WHITE {score[1]}'
        log.info(score_str)
        log.info('\n' + str(game))

    def play_match(self, n: int):
        log.info(f'Playing match: {n}')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        player1 = self.player1_class(
            Player.BLACK, *self.player1_args, **self.player1_kwargs)
        player2 = self.player2_class(
            Player.WHITE, *self.player2_args, **self.player2_kwargs)
        one_won = 0
        two_won = 0
        draws = 0
        tolerance = 0.01

        game, _ = Game.run_game_new(player1, player2, is_verbose=self.verbose)
        game_result = game.get_rewards(game.get_score())[0]
        if game_result > tolerance:
            one_won += 1
        elif game_result < -tolerance:
            two_won += 1
        else:
            draws += 1
        self.print_game_result(game)

        player1.player = Player.WHITE
        player2.player = Player.BLACK
        game, _ = Game.run_game_new(player2, player1, is_verbose=self.verbose)
        game_result = game.get_rewards(game.get_score())[0]

        if game_result > tolerance:
            two_won += 1
        elif game_result < -tolerance:
            one_won += 1
        else:
            draws += 1
        self.print_game_result(game)
        return (one_won, two_won, draws)

    def player_games(self):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            one_won: games won by player1
            two_won: games won by player2
            draws:  games won by nobody
        """

        with mp.Pool(processes=self.args.num_arena_workers) as pool:
            log.info('starting matches')
            scores = pool.map(
                self.play_match,
                range(0, self.args.num_arena_comparisons)
            )

            result = (0, 0, 0)
            for score in scores:
                result = np.add(result, score)
            return result
