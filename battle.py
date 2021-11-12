import coloredlogs
from abalone_engine.players import AlphaBetaPlayer

from src.Abalone import AbaloneNNPlayer
from src.tournament import run_tournament

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.
run_tournament([AlphaBetaPlayer, AbaloneNNPlayer])
