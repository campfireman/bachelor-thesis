import hashlib
import logging
import math
import time
from typing import List

import numpy as np
import numpy.typing as npt
from alpha_zero_general.Game import Game

from src.neural_net import NNetWrapper
from src.settings import CoachArguments

EPS = 1e-8

log = logging.getLogger(__name__)
# TIMES_TOTAL = []
# TIMES_PREDICT = []
# TIMES_LEGAL = []


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game: Game, nnet: NNetWrapper, args: CoachArguments):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.init_stores()

    def init_stores(self):
        self.Q_sa = {}  # stores Q values for s,a (as defined in the paper)
        self.N_sa = {}  # stores #times edge s,a was visited
        self.N_s = {}  # stores #times board s was visited
        self.P_s = {}  # stores initial policy (returned by neural net)

        self.R_s = {}  # stores game.get_game_ended ended for board s
        self.V_s = {}  # stores game.get_valid_moves for board s

    def get_action_prob(self, canonicalBoard: npt.NDArray, temp: int = 1) -> List[float]:
        """
        This function performs num_MCTS_sims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        self.init_stores()
        # times = []
        for i in range(self.args.num_MCTS_sims):
            # start = time.time()
            self.search(canonicalBoard)
            # end = time.time() - start
            # times.append(end)
        # TIMES_TOTAL.extend(times)
        # if len(TIMES_TOTAL) == 3000:
        #     print(f'+++ RESULT +++')
        # else:
        #     print(len(TIMES_TOTAL))

        # print(f'time per search: {np.average(TIMES_TOTAL)}')
        # print(f'time per predict: {np.average(TIMES_PREDICT)}')
        # print(f'time per legal: {np.average(TIMES_LEGAL)}')

        s = self.hash_state(canonicalBoard, '', 0)
        counts = [self.N_sa[(s, a)] if (
            s, a) in self.N_sa else 0 for a in range(self.game.get_action_size())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def hash_state(self, board: npt.NDArray, parent_hash: str, depth: int) -> str:
        hash = self.game.string_representation(
            board) + hashlib.md5((parent_hash + str(depth)).encode()).hexdigest()
        return hash

    def search(self, canonicalBoard: npt.NDArray, prev: str = '', depth: int = 0) -> float:
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.hash_state(canonicalBoard, prev, depth)

        if s not in self.R_s:
            self.R_s[s] = self.game.get_game_ended(canonicalBoard, 1)
        if self.R_s[s] != 0:
            # terminal node
            return -self.R_s[s]

        if s not in self.P_s:
            # leaf node
            # start = time.time()
            self.P_s[s], v = self.nnet.predict(canonicalBoard)
            # end = time.time() - start
            # TIMES_PREDICT.append(end)
            # start = time.time()
            valids = self.game.get_valid_moves(canonicalBoard, 1)
            # end = time.time() - start
            # TIMES_LEGAL.append(end)
            self.P_s[s] = self.P_s[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.P_s[s])
            if sum_Ps_s > 0:
                self.P_s[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                self.P_s[s] = self.P_s[s] + valids
                self.P_s[s] /= np.sum(self.P_s[s])

            self.V_s[s] = valids
            self.N_s[s] = 0
            return -v

        valids = self.V_s[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.get_action_size()):
            if valids[a]:
                if (s, a) in self.Q_sa:
                    u = self.Q_sa[(s, a)] + self.args.cpuct * self.P_s[s][a] * math.sqrt(self.N_s[s]) / (
                        1 + self.N_sa[(s, a)])
                else:
                    u = self.args.cpuct * \
                        self.P_s[s][a] * \
                        math.sqrt(self.N_s[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.get_next_state(canonicalBoard, 1, a)
        next_s = self.game.get_canonical_form(next_s, next_player)

        v = self.search(next_s, prev=s, depth=depth + 1)

        if (s, a) in self.Q_sa:
            self.Q_sa[(s, a)] = (self.N_sa[(s, a)] *
                                 self.Q_sa[(s, a)] + v) / (self.N_sa[(s, a)] + 1)
            self.N_sa[(s, a)] += 1

        else:
            self.Q_sa[(s, a)] = v
            self.N_sa[(s, a)] = 1

        self.N_s[s] += 1
        return -v
