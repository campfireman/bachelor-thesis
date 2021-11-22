import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CoachArguments:
    tpu_name: str = ''
    bucket_name: str = 'balthasar'
    num_iters: int = 1000
    num_eps: int = 2
    temp_treshhold: int = 15
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    update_treshold: float = 0.6
    # Max number of game examples in the queue between coach and self play workers
    maxlen_of_queue: int = 2000
    # Number of games moves for MCTS to simulate.
    num_MCTS_sims: int = 4
    cpuct: float = 1
    num_self_play_workers: int = 4
    # Number of games to play during arena play to determine if new net will be accepted.
    num_self_comparisons: int = 2
    # At which interval playoffs against heuristic agent and random agent are being performed
    agent_comparisons_step_size: int = 5
    num_random_agent_comparisons: int = 2
    num_heuristic_agent_comparisons: int = 2
    num_arena_workers: int = 2

    data_directory: str = './data'
    load_model: bool = False
    load_folder_file: Tuple[str, str] = (
        '/home/ture/projects/bachelor-thesis/code/src/temp', 'temp.pth.tar')
    num_iters_for_train_examples_history: int = 20

    @property
    def checkpoint(self) -> str:
        return os.path.join(self.data_directory, 'temp')

    # neural net arguments
    lr: float = 0.001
    dropout: float = 0.3
    epochs: int = 10
    batch_size: int = 64
    num_channels: int = 512
    residual_tower_size: int = 6
