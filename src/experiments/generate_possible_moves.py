from typing import List, Tuple, Union

from abalone_engine.enums import Direction, Space
from abalone_engine.game import Move
from abalone_engine.utils import neighbor


class BijectiveMap(dict):
    """
    thanks to https://archive.ph/jtAQy
    """

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        return dict.__len__(self) // 2


def is_possible_inline(marble: Space, direction: Direction) -> bool:
    if isinstance(marble, Space):
        if neighbor(marble, direction) is Space.OFF:
            return False
        return True
    return False


def is_possible_broadside(marbles: Tuple[Space, Space], marble_direction: Direction, move_direction: Direction) -> bool:
    if isinstance(marbles, tuple):
        if marble_direction is move_direction:
            return False
        if marble_direction.opposite_direction() is move_direction:
            return False
        return True
    return False


def sort_marbles(marbles: Tuple[Space, Space]) -> Tuple[Space, Space]:
    marble1 = ''.join(marbles[0].value)
    marble2 = ''.join(marbles[1].value)
    if marble1 < marble2:
        return marbles
    elif marble1 == marble2:
        raise ValueError('Marbles are equal!')
    else:
        return (marbles[1], marbles[0])


def create_lines(trailing_marble: Space, line_direction: Direction) -> List[Tuple[Space, Space]]:
    lines = []
    neighbor1 = neighbor(trailing_marble, line_direction)
    if neighbor1 is Space.OFF:
        return lines
    lines.append(sort_marbles((trailing_marble, neighbor1)))
    neighbor2 = neighbor(neighbor1, line_direction)
    if neighbor2 is Space.OFF:
        return lines
    lines.append(sort_marbles((trailing_marble, neighbor2)))
    return lines


def add_move(moves: BijectiveMap, counter: int, marbles: Union[Space, Tuple[Space, Space]], direction: Direction) -> int:
    move = Move.from_original((marbles, direction)).to_standard()
    if move in moves:
        return counter
    moves[move] = counter
    counter += 1
    return counter


def main():
    moves = BijectiveMap()
    counter = 0
    for space in Space:
        if space is Space.OFF:
            continue
        for direction in Direction:
            # in-line moves
            if is_possible_inline(space, direction):
                counter = add_move(moves, counter, space, direction)
            # broadside moves
            for line in create_lines(space, direction):
                for broadside_move_direction in Direction:
                    if is_possible_broadside(line, direction, broadside_move_direction):
                        counter = add_move(
                            moves, counter, line, broadside_move_direction)

    print(moves)


if __name__ == '__main__':
    main()
