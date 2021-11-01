import os
import sys


def open_pipe(path):
    with open(path, 'r') as fifo:
        while True:
            data = fifo.read()
            if len(data) > 0:
                print(data)


if __name__ == '__main__':
    path = "/tmp/recieve"
    os.mkfifo(path)
    print("Pipe is created")
    try:
        open_pipe(path)
    except KeyboardInterrupt:
        print("Stopped")
    os.unlink(path)
