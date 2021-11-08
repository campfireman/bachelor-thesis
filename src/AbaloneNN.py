from typing import List

import tensorflow as tf
import tensorflow.keras as keras

from src.experiments.possible_moves import POSSIBLE_MOVES


class AbaloneNN():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.input_boards = keras.Input(
            shape=(self.board_x, self.board_y, 2), name='board')
        convolutional_block = self.convolutional_block(self.input_boards)
        residual_tower = self.residual_tower(convolutional_block)
        self.pi = self.policy_head(residual_tower)
        self.v = self.value_head(residual_tower)
        self.model = keras.Model(
            inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=keras.optimizers.Adam(args['lr']))

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
        x = keras.layers.Dense(1)(x)
        x = keras.activations.tanh(x)
        return x

    def policy_head(self, x):
        x = keras.layers.Conv2D(2, (1, 1))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(len(POSSIBLE_MOVES) // 2)(x)
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
