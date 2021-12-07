import json
import os
from dataclasses import asdict, dataclass
from typing import Tuple

from tensorflow.python.lib.io import file_io


@dataclass
class CoachArguments:
    tpu_name: str = ''
    bucket_name: str = 'balthasar'
    num_iters: int = 1000
    num_eps: int = 10
    temp_treshhold: int = 15
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    update_treshold: float = 0.6
    # Max number of game examples in the queue between coach and self play workers
    maxlen_of_queue: int = 2000
    # Number of games moves for MCTS to simulate.
    num_MCTS_sims: int = 30
    cpuct: float = 1
    num_self_play_workers: int = 4
    num_arena_workers: int = 2
    self_play_worker_cpu: bool = True
    arena_worker_cpu: bool = True
    filter_by_reward_threshold: bool = False
    reward_threshold: float = 0.001
    # Number of games to play during arena play to determine if new net will be accepted.
    num_self_comparisons: int = 4
    # At which interval playoffs against heuristic agent and random agent are being performed
    agent_comparisons_step_size: int = 5
    num_random_agent_comparisons: int = 4
    num_heuristic_agent_comparisons: int = 2

    data_directory_name: str = './data'
    load_model: bool = False
    load_folder_file: Tuple[str, str] = (
        '/home/ture/projects/bachelor-thesis/code/data/temp', 'best.pth.tar')
    num_iters_for_train_examples_history: int = 100

    # neural net arguments
    framework: str = 'torch'
    nnet_size: str = 'large'
    lr: float = 0.001
    dropout: float = 0.3
    epochs: int = 10
    batch_size: int = 64
    num_channels: int = 512
    residual_tower_size: int = 6
    cuda: bool = True

    @property
    def data_directory(self) -> str:
        if self.tpu_name:
            return os.path.join(
                'gs://', self.args.bucket_name, os.path.normpath(self.data_directory_name))
        return self.data_directory_name

    @property
    def checkpoint(self) -> str:
        return os.path.join(self.data_directory, 'temp')

    def save(self, timestamp: float):
        filepath = os.path.join(self.data_directory,
                                f'{timestamp}_settings.json')
        with file_io.FileIO(filepath, 'w') as file:
            file.write(json.dumps(asdict(self), indent=4))
