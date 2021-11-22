import coloredlogs
from abalone_engine.players import (AbaProPlayer, AlphaBetaPlayerFast,
                                    RandomPlayer)

from src.abalone_game import AbaloneNNPlayer
from src.tournament import run_tournament

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.
# run_tournament([AbaProPlayer, RandomPlayer])
run_tournament([AbaProPlayer, RandomPlayer])
