import logging
from typing import Iterable, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from abalone_engine.enums import Direction, Player, Space
from abalone_engine.game import Game as Engine
from abalone_engine.game import Move
from abalone_engine.players import AbstractPlayer
from alpha_zero_general.Game import Game

from src.experiments.possible_moves import POSSIBLE_MOVES
from src.mcts import MCTS
from src.neural_net import NNetWrapper as NNetWrapperTensorflow
from src.neural_net_torch import NNetWrapper as NNetWrapperTorch
from src.settings import CoachArguments

from .utils import move_index_to_standard, move_standard_to_index

MAX_TURNS = 200
TOTAL_NUM_MARBLES = 14
TAKEN_MARBLES_TO_WIN = 6


log = logging.getLogger(__name__)


class AbaloneGame(Game):
    """
    Based on https://github.com/suragnair/alpha-zero-general/blob/master/Game.py
    """

    def __init__(self):
        self.engine = Engine

    def get_init_board(self) -> npt.NDArray:
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return np.array([
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
        ], dtype='int8')

    def get_board_size(self) -> Tuple[int, int]:
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (9, 9)

    def get_action_size(self) -> int:
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 1452

    def get_next_state(self, board: npt.NDArray, player: int, action: int) -> npt.NDArray:
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        board = np.copy(board)
        next_player = -1 if player == 1 else 1
        board = self.engine.s_standard_move(
            board, player, move_index_to_standard(action))
        return board, next_player

    def get_valid_moves(self, board: npt.NDArray, player: int):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.get_action_size(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        moves = np.zeros(self.get_action_size(), dtype=np.float32)
        for move in self.engine.s_generate_legal_moves(board, player):
            index = move_standard_to_index(
                Move.from_original(move).to_standard())
            moves[index] = 1
        return moves

        # return s_get_legal_moves(board, player)

    def get_game_ended(self, board: npt.NDArray, player: int) -> int:
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        score = self.engine.s_score(board)
        if self.engine.s_is_over(score):
            return 1 if self.engine.s_winner(score).value == player else -1
        return 0

    def get_game_ended_limited(self, board: npt.NDArray, player: int, turns: int) -> float:
        if turns > MAX_TURNS:
            score = self.engine.s_score(board)
            marbles_taken_black = (TOTAL_NUM_MARBLES -
                                   score[1]) / TAKEN_MARBLES_TO_WIN
            marbles_taken_white = (TOTAL_NUM_MARBLES -
                                   score[0]) / TAKEN_MARBLES_TO_WIN
            partial_score = marbles_taken_black - \
                marbles_taken_white if player == 1 else marbles_taken_white - marbles_taken_black
            if partial_score == 0:
                partial_score = 1e-8
            log.info(
                f'Exceeded max turns of {MAX_TURNS} setting game result to {partial_score}')
            return partial_score
        return self.get_game_ended(board, player)

    def get_canonical_form(self, board: npt.NDArray, player: int) -> npt.NDArray:
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return player*board

    def get_symmetries(self, board: npt.NDArray, pi: List[float]) -> Iterable[Tuple[npt.NDArray, List[float]]]:
        """
        Input:
            board: current board
            pi: policy vector of size self.get_action_size()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        # assume player 1 as not relevant for symmetries
        game = self.engine.from_array(board, 1)
        moves = []
        # already include 360 degree rotation
        yield board, pi

        for i in range(0, len(pi)):
            if pi[i] != 0:
                moves.append({
                    'move': Move.from_standard(
                        POSSIBLE_MOVES[i]),
                    'value': pi[i],
                })

        store = {}
        for deg in range(60, 360, 60):
            rotated_board = game.to_rotated_array(deg)
            s = self.string_representation(rotated_board)
            if s in store:
                continue
            store[s] = True

            rotated_pi = np.zeros(self.get_action_size())
            # convert move to rotated move, rotated move to index
            for move in moves:
                rotated_move_index = move_standard_to_index(
                    move['move'].rotate(deg).to_standard())
                rotated_pi[rotated_move_index] = move['value']
            yield rotated_board, rotated_pi

        for axis in ('q', 'qx', 'r', 'rx', 's', 'sx'):
            reflected_board = game.to_reflected_array(axis)
            s = self.string_representation(reflected_board)
            if s in store:
                continue
            store[s] = True
            reflected_pi = np.zeros(self.get_action_size())
            # convert board to rotated board
            for move in moves:
                reflected_move_index = move_standard_to_index(
                    move['move'].reflect(axis).to_standard())
                reflected_pi[reflected_move_index] = move['value']
            yield reflected_board, reflected_pi

    def string_representation(self, board: npt.NDArray) -> str:
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        string = ''
        for line in board:
            string += ''.join(map(str, line))
        return string

    def display_state(self, board: npt.NDArray):
        game = self.engine.from_array(board, player=1)
        score = game.get_score()
        score_str = f'BLACK {score[0]} - WHITE {score[1]}'
        print(score_str, game, '', sep='\n')


class AbaloneNNPlayer(AbstractPlayer):
    def __init__(self, player: Player, nnet_fullpath: str, args: CoachArguments):
        super().__init__(player)
        self.game = AbaloneGame()
        self.args = args
        self.model = self.load_model(nnet_fullpath)
        self.mcts = MCTS(self.game, self.model, self.args)

    def load_model(self, nnet_fullpath):
        if self.args.framework == 'tensorflow':
            nn = NNetWrapperTensorflow(self.game, self.args)
        elif self.args.framework == 'torch':
            nn = NNetWrapperTorch(self.game, self.args)
        else:
            raise ValueError('Unknown ML framework')
        nn.load_checkpoint(full_path=nnet_fullpath)
        return nn

    def search(self, board: np.array) -> int:
        return np.argmax(self.mcts.get_action_prob(board, temp=0))

    def turn(self, game: Game, moves_history: List[Tuple[Union[Space, Tuple[Space, Space]], Direction]]) -> Tuple[Union[Space, Tuple[Space, Space]], Direction]:
        board = game.canonical_board()
        return Move.from_standard(move_index_to_standard(self.search(board))).to_original()
