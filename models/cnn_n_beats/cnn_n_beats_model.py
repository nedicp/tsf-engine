import tensorflow as tf
from .cnn_n_beats_block import NbeatsBlock


def create_cnn_NBEATS(config):
    """
    Creates a CNN-N-BEATS model with the specified configuration.

    Args:
        config: Configuration dictionary containing:
            - input_shape: Shape of the input sequence
            - horizon: Number of future timesteps to predict
            - n_neurons: Number of neurons in hidden layers
            - l2_reg: L2 regularization factor
            - num_blocks: Number of blocks
            - dropout_rate: Dropout rate
            - n_layers: Number of layers

    Returns:
        tf.keras.Model: Compiled CNN-N-BEATS model
    """

    input_shape = config["model_params"]["input_shape"]
    horizon = config["model_params"]["horizon"]
    n_neurons = config["model_params"]["n_neurons"]
    l2_reg = config["model_params"]["l2_reg"]
    num_blocks = config["model_params"]["num_blocks"]
    dropout_rate = config["model_params"]["dropout_rate"]
    n_layers = config["model_params"]["n_layers"]

    def subtract_last_value(x):
        last_value = x[:, -1, -1]
        last_value_expanded = tf.expand_dims(
            tf.expand_dims(last_value, axis=-1), axis=-1
        )
        mask = tf.concat(
            [tf.zeros_like(x[..., :-1]), tf.ones_like(x[..., -1:])], axis=-1
        )
        modified_x = x - mask * last_value_expanded
        return modified_x, tf.expand_dims(last_value, axis=-1)

    def add_last_value_to_forecast(forecast_last_value):
        forecast, last_value = forecast_last_value
        last_value_expanded = tf.expand_dims(last_value, axis=-1)
        last_value_expanded = tf.tile(last_value_expanded, [1, horizon, 1])
        last_value_expanded = tf.squeeze(last_value_expanded, axis=-1)
        return forecast + last_value_expanded

    stack_input = tf.keras.layers.Input(shape=input_shape)
    processed_input, last_value = tf.keras.layers.Lambda(subtract_last_value)(
        stack_input
    )

    x = processed_input
    x = tf.keras.layers.Conv1D(
        filters=n_neurons,
        kernel_size=3,
        padding="same",
        activation="relu",
        dilation_rate=1,
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
    )(x)

    x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid")(x)
    x = tf.keras.layers.Flatten()(x)

    i_flat = tf.keras.layers.Flatten()(processed_input)
    x = tf.keras.layers.Concatenate(axis=-1)([x, i_flat])
    x = tf.keras.layers.Dense(
        n_neurons,
        activation="linear",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
    )(x)
    
    forecast, backcast = NbeatsBlock(
        input_size=n_neurons,  # because of the Dense layer above.
        dropout_rate=dropout_rate,
        horizon=horizon,
        neurons=n_neurons,
        layers=n_layers,
        l2_reg=l2_reg,
        name="TrendBlock0",
    )(x)
    
    residuals = tf.keras.layers.subtract([x, backcast])

    for i in range(num_blocks - 1):
        block_forecast, backcast = NbeatsBlock(
            input_size=n_neurons,
            dropout_rate=dropout_rate,
            horizon=horizon,
            neurons=n_neurons,
            layers=n_layers,
            l2_reg=l2_reg,
            name=f"TrendBlock{i + 1}",
        )(residuals)
        residuals = tf.keras.layers.subtract([residuals, backcast])
        forecast = tf.keras.layers.add([forecast, block_forecast])

    forecast = tf.keras.layers.Lambda(
        add_last_value_to_forecast, output_shape=(horizon,)
    )([forecast, last_value])

    return tf.keras.Model(inputs=stack_input, outputs=forecast)