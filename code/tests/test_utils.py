import os

from src import utils


def test_csv_table():
    path = os.path.dirname(os.path.realpath(__file__))
    filename = 'test.csv'
    header = ['Test', 'Test 2', 'Test 3']
    table = utils.CsvTable(path, filename, header)
    table.add_row(['test', 'test', 'test'])
    os.unlink(os.path.join(path, filename))
