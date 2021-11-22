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
        self.filepath = os.path.join(path, filename)
        self.header = header
        self.add_row(self.header)

    def add_row(self, row: List):
        with open(self.filepath, 'a+') as file:
            if len(row) != len(self.header):
                raise ArgumentError(
                    f'Incorrect dimension of row, header has {len(self.header)} and row has {len(row)}')
            csv_writer = csv.writer(file, delimiter=',')
            csv_writer.writerow(row)
