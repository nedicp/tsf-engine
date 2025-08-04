import yaml
from pathlib import Path
from models.cnn_n_beats.cnn_n_beats_model import create_cnn_NBEATS
from models.nbeats.nbeats_model import create_NBEATS_sin_cos
from models.nbeats_cnn.nbeats_cnn_model import create_NBEATS_CNN

def load_model_config(model: str):
    config_path = Path("models/configs/model_config.yaml")
    file = open(config_path, "r")
    config = yaml.load(file, Loader=yaml.FullLoader)
    file.close()
    return config[model]

def load_models_state_level_24h():
    """
    Loads all available models for state level prediction with 24h horizon.
    Returns:
        models (tf.keras.Model): Loaded models
    """

    config = load_model_config("state_level_24h")

    print(config["nbeats_cnn"])

    model_nbeats_cnn = create_NBEATS_CNN(config=config["nbeats_cnn"])
    model_nbeats_cnn.load_weights(Path("models/weights/state_level_24h/model_conv.weights.h5"))

    model_cnn_nbeats = create_cnn_NBEATS(config=config["cnn_nbeats"])
    model_cnn_nbeats.load_weights(Path("models/weights/state_level_24h/model_cnn_nbeats.weights.h5"))

    model_nbeats = create_NBEATS_sin_cos(config=config["nbeats"])
    model_nbeats.load_weights(Path("models/weights/state_level_24h/model_nbeats.weights.h5"))

    return model_nbeats_cnn, model_cnn_nbeats, model_nbeats
