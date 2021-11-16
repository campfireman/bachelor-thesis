import os
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from alpha_zero_general.NeuralNet import NeuralNet
from alpha_zero_general.utils import dotdict

from .experiments.possible_moves import POSSIBLE_MOVES

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
    'residual_tower_size': 6,
})


class AbaloneNN():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.input_boards = keras.Input(
            shape=(self.board_x, self.board_y), name='board')
        board = keras.layers.Reshape(
            (self.board_x, self.board_y, 1))(self.input_boards)
        convolutional_block = self.convolutional_block(board)
        residual_tower = self.residual_tower(
            convolutional_block, size=args.residual_tower_size)
        self.pi = self.policy_head(residual_tower)
        self.v = self.value_head(residual_tower)
        self.model = keras.Model(
            inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=keras.optimizers.Adam(args.lr))

    def convolutional_block(self, x):
        x = keras.layers.Conv2D(256, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        return x

    def residual_tower(self, x, size: int = 19):
        for i in range(0, size):
            x_input = x
            x = self.convolutional_block(x)
            x = keras.layers.Conv2D(256, (3, 3), padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            # skip connection
            x = keras.layers.Add()([x_input, x])
            x = keras.layers.ReLU()(x)
        return x

    def value_head(self, x):
        x = keras.layers.Conv2D(2, (1, 1))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dense(256)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1, activation='tanh', name='v')(x)
        return x

    def policy_head(self, x):
        x = keras.layers.Conv2D(2, (1, 1))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(len(POSSIBLE_MOVES) // 2,
                               activation='softmax', name='pi')(x)
        return x

    def trace(self):
        @tf.function
        def traceme(x):
            return self.model(x)

        logdir = "log"
        writer = tf.summary.create_file_writer(logdir)
        tf.summary.trace_on(graph=True, profiler=True)
        # Forward pass
        traceme(tf.zeros((1, 9, 9, 2)))
        with writer.as_default():
            tf.summary.trace_export(
                name="model_trace", step=0, profiler_outdir=logdir)

    def visualize(self):
        from tensorflow.keras.utils import plot_model
        plot_model(self.model, to_file='abalone_NN_model.png',
                   show_shapes=True)

    def show_info(self):
        self.model.summary()


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = AbaloneNN(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x=input_boards, y=[
                            target_pis, target_vs], batch_size=args.batch_size, epochs=args.epochs)

    def predict(self, board):
        """
        board: np array with board
        """
        # preparing input
        board = board[np.newaxis, :, :]
        pi, v = self.nnet.model.predict(board)
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        self.nnet.model = tf.keras.models.load_model(filepath)
