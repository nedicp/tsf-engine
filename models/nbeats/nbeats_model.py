import tensorflow as tf
import numpy as np
from .nbeats_blocks import NBeatsBlockSeason, NBeatsBlockTrend

def create_NBEATS_sin_cos(config) -> tf.keras.Model:
    """
    Creates an N-BEATS model with trend and seasonal blocks.

    Args:
        config: Configuration dictionary containing:
            - input_shape: Shape of the input sequence
            - horizon: Number of future timesteps to predict
            - n_neurons: Number of neurons in hidden layers
            - n_layers: Number of hidden layers
            - poly_order: Order of polynomial for trend block
            - l2_reg: L2 regularization factor
            - dropout_rate: Dropout rate
            - num_blocks_trend: Number of trend blocks
            - num_blocks_seasonal: Number of seasonal blocks

    Returns:
        tf.keras.Model: Compiled N-BEATS model
    """
    
    input_shape = config["model_params"]["input_shape"]
    horizon = config["model_params"]["horizon"]
    n_neurons = config["model_params"]["n_neurons"]
    n_layers = config["model_params"]["n_layers"]
    poly_order = config["model_params"]["poly_order"]
    l2_reg = config["model_params"]["l2_reg"]
    dropout_rate = config["model_params"]["dropout_rate"]
    num_blocks_trend = config["model_params"]["num_blocks_trend"]
    num_blocks_seasonal = config["model_params"]["num_blocks_seasonal"]
    
    stack_input = tf.keras.layers.Input(shape=input_shape)

    input_size = np.prod(input_shape)
    
    x = (
        tf.keras.layers.Flatten()(stack_input) if len(input_shape) > 1 else stack_input
    )

    backcast, forecast = NBeatsBlockTrend(
        input_size=input_size,
        dropout_rate=dropout_rate,
        horizon=horizon,
        neurons=n_neurons,
        layers=n_layers,
        poly_order=poly_order,
        l2_reg=l2_reg,
        name="TrendBlock0",
    )(x)

    residuals = tf.keras.layers.subtract([x, backcast])

    for i in range(num_blocks_trend - 1):
        backcast, block_forecast = NBeatsBlockTrend(
            input_size=input_size,
            dropout_rate=dropout_rate,
            horizon=horizon,
            neurons=n_neurons,
            layers=n_layers,
            poly_order=poly_order,
            l2_reg=l2_reg,
            name=f"TrendBlock{i + 1}",
        )(residuals)
        residuals = tf.keras.layers.subtract([residuals, backcast])
        forecast = tf.keras.layers.add([forecast, block_forecast])

    for i in range(num_blocks_seasonal):
        backcast, block_forecast = NBeatsBlockSeason(
            input_size=input_size,
            dropout_rate=dropout_rate,
            horizon=horizon,
            neurons=n_neurons,
            layers=n_layers,
            l2_reg=l2_reg,
            name=f"SeasonalBlock{i}",
        )(residuals)
        residuals = tf.keras.layers.subtract([residuals, backcast])
        forecast = tf.keras.layers.add([forecast, block_forecast])

    model = tf.keras.Model(inputs=stack_input, outputs=forecast)
    return model