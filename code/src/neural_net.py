import os
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.backend import dtype

from src.abalone_game import Game
from src.settings import CoachArguments

from .experiments.possible_moves import POSSIBLE_MOVES


class AbaloneNNet():
    def __init__(self, game: Game, args: 'CoachArguments'):
        # game params
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
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

    def convolutional_block(self, x: tf.Tensor) -> tf.Tensor:
        x = keras.layers.Conv2D(256, (3, 3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        return x

    def residual_tower(self, x: tf.Tensor, size: int = 19) -> tf.Tensor:
        for i in range(0, size):
            x_input = x
            x = self.convolutional_block(x)
            x = keras.layers.Conv2D(256, (3, 3), padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            # skip connection
            x = keras.layers.Add()([x_input, x])
            x = keras.layers.ReLU()(x)
        return x

    def value_head(self, x: tf.Tensor) -> tf.Tensor:
        x = keras.layers.Conv2D(2, (1, 1))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dense(256)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1, activation='tanh', name='v')(x)
        return x

    def policy_head(self, x: tf.Tensor) -> tf.Tensor:
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


class AbaloneNNetMini():
    """
    Based on: https://github.com/suragnair/alpha-zero-general/blob/master/othello/tensorflow/OthelloNNet.py
    """

    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

        # Neural Net
        # s: batch_size x board_x x board_y
        self.input_boards = keras.layers.Input(
            shape=(self.board_x, self.board_y))

        x_image = keras.layers.Reshape((self.board_x, self.board_y, 1))(
            self.input_boards)                # batch_size  x board_x x board_y x 1
        h_conv1 = keras.layers.Activation('relu')(keras.layers.BatchNormalization(axis=3)(keras.layers.Conv2D(args.num_channels, 3, padding='same', use_bias=False)(
            x_image)))         # batch_size  x board_x x board_y x num_channels
        h_conv2 = keras.layers.Activation('relu')(keras.layers.BatchNormalization(axis=3)(keras.layers.Conv2D(args.num_channels, 3, padding='same', use_bias=False)(
            h_conv1)))         # batch_size  x board_x x board_y x num_channels
        h_conv3 = keras.layers.Activation('relu')(keras.layers.BatchNormalization(axis=3)(keras.layers.Conv2D(args.num_channels, 3, padding='same', use_bias=False)(
            h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = keras.layers.Activation('relu')(keras.layers.BatchNormalization(axis=3)(keras.layers.Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(
            h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = keras.layers.Flatten()(h_conv4)
        s_fc1 = keras.layers.Dropout(args.dropout)(keras.layers.Activation('relu')(keras.layers.BatchNormalization(axis=1)(
            keras.layers.Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = keras.layers.Dropout(args.dropout)(keras.layers.Activation('relu')(keras.layers.BatchNormalization(axis=1)(
            keras.layers.Dense(512, use_bias=False)(s_fc1))))          # batch_size x 1024
        self.pi = keras.layers.Dense(self.action_size, activation='softmax', name='pi')(
            s_fc2)   # batch_size x self.action_size
        self.v = keras.layers.Dense(1, activation='tanh', name='v')(
            s_fc2)                    # batch_size x 1

        self.model = keras.Model(
            inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy',
                           'mean_squared_error'], optimizer=keras.optimizers.Adam(args.lr))


class NNetWrapperBase():
    def train(self, examples: List[Tuple[npt.NDArray, List[float], float]]):
        raise NotImplementedError

    def predict_old(self, board: npt.NDArray) -> Tuple[npt.NDArray, float]:
        raise NotImplementedError

    def predict(self, board: npt.NDArray):
        raise NotImplementedError

    def save_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar'):
        raise NotImplementedError

    def load_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar', full_path: str = None):
        raise NotImplementedError

    def show_info(self):
        raise NotImplementedError


class NNetWrapper(NNetWrapperBase):
    def __init__(self, game, args: CoachArguments):
        if args.nnet_size == 'large':
            self.nnet = AbaloneNNet(game, args)
        else:
            self.nnet = AbaloneNNetMini(game, args)
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

    def train(self, examples: List[Tuple[npt.NDArray, List[float], float]]):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x=input_boards, y=[
                            target_pis, target_vs], batch_size=self.args.batch_size, epochs=self.args.epochs)

    def predict_old(self, board: npt.NDArray) -> Tuple[npt.NDArray, float]:
        # preparing input
        board = board[np.newaxis, :, :]
        pi, v = self.nnet.model.predict(board)
        return pi[0], v[0]

    def predict(self, board: npt.NDArray):
        """
        board: np array with board
        """
        board = board[np.newaxis, :, :]
        pi, v = self.nnet.model(
            board, training=False)
        return pi[0].numpy(), v[0].numpy()

    def save_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save(filepath)

    def load_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar', full_path: str = None):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(
            folder, filename) if full_path is None else full_path
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        self.nnet.model = tf.keras.models.load_model(filepath)

    def show_info(self):
        self.nnet.model.summary()
