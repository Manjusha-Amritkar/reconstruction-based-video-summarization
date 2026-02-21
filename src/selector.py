"""
BiLSTM Selector Network
Predicts frame-level importance scores
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional


def build_selector(input_dim=2048, hidden_dim=512):
    """
    Builds BiLSTM-based selector model.

    Args:
        input_dim (int): Feature dimension
        hidden_dim (int): Hidden LSTM units

    Returns:
        tf.keras.Model
    """

    inputs = Input(shape=(None, input_dim))

    lstm_out = Bidirectional(
        LSTM(hidden_dim, return_sequences=True)
    )(inputs)

    scores = Dense(1, activation="sigmoid")(lstm_out)

    model = Model(inputs, scores, name="BiLSTM_Selector")

    return model