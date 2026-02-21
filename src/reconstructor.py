"""
BiLSTM Encoder–Decoder Reconstructor Model
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional


def build_reconstructor(input_dim=2048, hidden_dim=512):
    """
    Builds a 4-layer BiLSTM encoder–decoder model.

    Args:
        input_dim (int): Feature dimension (default: 2048)
        hidden_dim (int): Hidden LSTM units

    Returns:
        tf.keras.Model
    """

    inputs = Input(shape=(None, input_dim))

    # Encoder
    encoded = Bidirectional(
        LSTM(hidden_dim, return_sequences=True)
    )(inputs)

    encoded = Bidirectional(
        LSTM(hidden_dim // 2, return_sequences=True)
    )(encoded)

    # Decoder
    decoded = Bidirectional(
        LSTM(hidden_dim // 2, return_sequences=True)
    )(encoded)

    decoded = Bidirectional(
        LSTM(hidden_dim, return_sequences=True)
    )(decoded)

    # Reconstruction output
    outputs = Dense(input_dim, activation="linear")(decoded)

    model = Model(inputs, outputs, name="BiLSTM_Reconstructor")

    return model