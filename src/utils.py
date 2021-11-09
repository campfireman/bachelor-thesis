from .experiments.possible_moves import POSSIBLE_MOVES


def move_index_to_standard(index: int) -> str:
    return POSSIBLE_MOVES[index]


def move_standard_to_index(move: str) -> int:
    return POSSIBLE_MOVES[move]
