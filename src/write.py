import argparse
import os
import sys


def write_pipe(path, msg):
    with open(path, 'w') as fifo:
        data = fifo.write(msg)
    print("Message sent")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('message', type=str)
    args = parser.parse_args()

    path = "/tmp/recieve"
    write_pipe(path, args.message)
