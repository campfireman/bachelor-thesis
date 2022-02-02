# Bachelor thesis 2021

## Abstract

AlphaGo's victory against Lee Sedol in the game of Go has been a milestone in artificial intelligence. After this success, the team behind the program further refined the architecture and applied it to many other games such as chess or shogi. In the following thesis, we try to apply the theory behind AlphaGo and its successor AlphaZero to the game of Abalone. Due to limitations in computational resources, we could not replicate the same exceptional performance.

## Run training pipeline

```[bash]
./install_ubuntu.sh
```

If one a different OS, make sure you have CUDA and CUDNN installed. To install the Python dependencies run:

```[bash]
pip3 install numpy
pip3 install Cython
pip3 install -r requirements.txt
```

To exectue the training pipeline with hyperparameters defined in the file `test.json` (while in root of the project):

```[bash]
python3 train.py --args test.json
```
