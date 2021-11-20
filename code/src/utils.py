import csv
import os
from ctypes import ArgumentError
from typing import List

from .experiments.possible_moves import POSSIBLE_MOVES


def move_index_to_standard(index: int) -> str:
    return POSSIBLE_MOVES[index]


def move_standard_to_index(move: str) -> int:
    return POSSIBLE_MOVES[move]


class CsvTable:
    def __init__(self, path: str, filename: str, header: List):
        self.file = open(os.path.join(path, filename), 'w')
        self.csv_writer = csv.writer(self.file, delimiter=',')
        self.header = header
        self.csv_writer.writerow(self.header)

    def add_row(self, row: List):
        if len(row) != len(self.header):
            raise ArgumentError(
                f'Incorrect dimension of row, header has {len(self.header)} and row has {len(row)}')
        self.csv_writer.writerow(row)

    def __del__(self):
        self.file.close()
