import sys  # added!

sys.path.append("..")  # added!

from multiprocessing import Pool

from abalone_engine.enums import Player
from abalone_engine.game import Game, Move
from abalone_engine.players import RandomPlayer
from src.experiments.possible_moves import POSSIBLE_MOVES


def run_and_check(i):
    print(f'Running game {i}')
    game, move_history, game_stats = Game.run_game(
        RandomPlayer(Player.BLACK), RandomPlayer(Player.WHITE), is_verbose=False)
    print(f'Validating game {i}')
    for move in move_history:
        POSSIBLE_MOVES[Move.from_original(move).to_standard()]

def main():
    with Pool() as p:
        p.map(run_and_check, range(0, 100))


if __name__ == '__main__':
    main()
