import json

from models.cnn_n_beats.cnn_n_beats_model import create_cnn_NBEATS
from models.nbeats.nbeats_model import create_NBEATS_sin_cos
from models.nbeats_cnn.nbeats_cnn_model import create_NBEATS_CNN

def load_models():
    """
    Loads all models.
    Returns:
        models (tf.keras.Model): Loaded models
    """

    file = open("models/nbeats_cnn/config.json", "r")
    config = json.load(file)
    file.close()
    model_nbeats_cnn = create_NBEATS_CNN(config=config)
    model_nbeats_cnn.load_weights("models/nbeats_cnn/model_conv.weights.h5")
    
    file = open("models/cnn_n_beats/config.json", "r")
    config = json.load(file)
    file.close()
    model_cnn_nbeats = create_cnn_NBEATS(config=config)
    model_cnn_nbeats.load_weights("models/cnn_n_beats/model_cnn_nbeats.weights.h5")

    file = open("models/nbeats/config.json", "r")
    config = json.load(file)
    file.close()
    model_nbeats = create_NBEATS_sin_cos(config=config)
    model_nbeats.load_weights("models/nbeats/model_nbeats.weights.h5")
    
    return model_nbeats_cnn, model_cnn_nbeats, model_nbeats


