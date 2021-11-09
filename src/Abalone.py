import numpy as np
from abalone_engine.game import Game as Engine
from abalone_engine.game import Move
from alpha_zero_general import Game

from .utils import move_index_to_standard, move_standard_to_index


class AbaloneGame(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """

    def __init__(self):
        self.engine = Engine

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return [
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
        ]

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (9, 9)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 1452

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        game = self.engine.from_array(board, player)
        game.standard_move(move_index_to_standard(action))
        return game.to_array()

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        game = self.engine.from_array(board, player)
        moves = np.zeros(self.getActionSize(), dtype=np.float32)
        for move in game.generate_legal_moves():
            index = move_standard_to_index(
                Move.from_original(move).to_standard())
            moves[index] = 1
        return moves

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        game = self.engine.from_array(board, player)
        if game.is_over():
            return 1 if game.get_winner().value == player else -1
        return 0

    def getCanonicalForm(self, board, player):
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
        return self.engine.from_array(board, player).canonical_board()

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(board, pi)]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        string = ''
        for line in board:
            string.append(''.join(line))
        return string
