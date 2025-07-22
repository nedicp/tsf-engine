import tensorflow as tf


class NbeatsBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_size: int,
        dropout_rate: float,
        horizon: int,
        neurons: int,
        layers: int,
        l2_reg: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.horizon = horizon
        self.neurons = neurons
        self.layers = layers

        self.hidden = [
            tf.keras.layers.Dense(
                neurons,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            )
            for _ in range(layers)
        ]
        self.theta_layer = tf.keras.layers.Dense(
            self.horizon + self.input_size, activation="linear"
        )
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs):
        x = inputs

        for layer in self.hidden:
            x = layer(x)

        x = self.dropout(x)

        theta = self.theta_layer(x)

        forecast = theta[:, : self.horizon]
        backcast = theta[:, self.horizon :]

        backcast = tf.reshape(backcast, tf.shape(inputs))

        return forecast, backcast
