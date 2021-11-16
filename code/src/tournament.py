import logging
from itertools import product
from typing import List, Tuple

from abalone_engine.enums import Player
from abalone_engine.game import Game
from abalone_engine.players import AbstractPlayer
from abalone_engine.utils import MoveStats

log = logging.getLogger(__name__)


def run_tournament(players: List[AbstractPlayer]) -> List[Tuple[Game, List, MoveStats]]:
    stats = []
    for black, white in product(players, repeat=2):
        if black == white:
            continue
        log.info(
            f'Playing {black.__name__} (Black) against {white.__name__} (White)')
        game, moves_history, move_stats = Game.run_game(
            black(Player.BLACK), white(Player.WHITE), is_verbose=True)
        stats.append((game, moves_history, move_stats))
    return stats
