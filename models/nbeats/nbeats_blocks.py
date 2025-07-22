import tensorflow as tf
import numpy as np

class NBeatsBlockSeason(tf.keras.layers.Layer):
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

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
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

        self.t_forecast = np.arange(horizon) / horizon
        self.t_backcast = np.arange(input_size) / (input_size)

    def call(self, inputs):
        x = inputs
        
        for layer in self.hidden:
            x = layer(x)

        if self.dropout_rate != 0:
            x = self.dropout(x)

        theta = self.theta_layer(x)
        theta_f = theta[:, : self.horizon]
        theta_b = theta[:, self.horizon : self.input_size + self.horizon]

        H = self.horizon
        B = self.input_size

        t_forecast = tf.convert_to_tensor(self.t_forecast, dtype=tf.float32)
        t_backcast = tf.convert_to_tensor(self.t_backcast, dtype=tf.float32)

        cos_terms_forecast = tf.transpose(
            tf.stack(
                [tf.cos(2 * np.pi * i * t_forecast) for i in range(H // 2)], axis=1
            )
        )
        sin_terms_forecast = tf.transpose(
            tf.stack(
                [tf.sin(2 * np.pi * i * t_forecast) for i in range(H // 2)], axis=1
            )
        )

        cos_terms_backcast = tf.transpose(
            tf.stack(
                [tf.cos(2 * np.pi * i * t_backcast) for i in range(B // 2)], axis=1
            )
        )
        sin_terms_backcast = tf.transpose(
            tf.stack(
                [tf.sin(2 * np.pi * i * t_backcast) for i in range(B // 2)], axis=1
            )
        )

        theta_cos_forecast = theta_f[:, : H // 2]
        theta_sin_forecast = theta_f[:, H // 2 : H]

        theta_cos_backcast = theta_b[:, : B // 2]
        theta_sin_backcast = theta_b[:, B // 2 : B]

        sum_cos_forecast = tf.matmul(theta_cos_forecast, cos_terms_forecast)
        sum_sin_forecast = tf.matmul(theta_sin_forecast, sin_terms_forecast)

        sum_cos_backcast = tf.matmul(theta_cos_backcast, cos_terms_backcast)
        sum_sin_backcast = tf.matmul(theta_sin_backcast, sin_terms_backcast)

        output_forecast = sum_cos_forecast + sum_sin_forecast
        output_backcast = sum_cos_backcast + sum_sin_backcast

        backcast, forecast = output_backcast, output_forecast
        backcast = tf.reshape(backcast, tf.shape(inputs))

        return backcast, forecast


class NBeatsBlockTrend(tf.keras.layers.Layer):
    def __init__(
        self,
        input_size: int,
        dropout_rate: float,
        horizon: int,
        neurons: int,
        layers: int,
        poly_order: int,
        l2_reg: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.horizon = horizon
        self.neurons = neurons
        self.layers = layers
        self.poly_order = poly_order

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.hidden = [
            tf.keras.layers.Dense(
                neurons,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            )
            for _ in range(layers)
        ]

        self.theta_layer = tf.keras.layers.Dense(
            (poly_order + 1) * 2, activation="linear"
        )

        t_forecast = np.arange(self.horizon) / self.horizon
        t_backcast = np.arange(self.input_size) / self.input_size

        self.T_forecast = np.column_stack(
            [t_forecast**i for i in range(poly_order + 1)]
        )
        self.T_backcast = np.column_stack(
            [t_backcast**i for i in range(poly_order + 1)]
        )

    def call(self, inputs):
        x = inputs

        for layer in self.hidden:
            x = layer(x)
            
        if self.dropout_rate != 0:
            x = self.dropout(x)

        theta = self.theta_layer(x)

        theta_f = theta[:, : self.poly_order + 1]
        theta_b = theta[:, self.poly_order + 1 :]

        T_forecast_tf = tf.transpose(
            tf.convert_to_tensor(self.T_forecast, dtype=tf.float32)
        )
        T_backcast_tf = tf.transpose(
            tf.convert_to_tensor(self.T_backcast, dtype=tf.float32)
        )

        forecast = tf.matmul(theta_f, T_forecast_tf)

        backcast = tf.matmul(theta_b, T_backcast_tf)
        backcast = tf.reshape(backcast, tf.shape(inputs))

        return backcast, forecast