import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input

def build_lstm_autoencoder(seq_len=24, n_features=2):
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(64, return_sequences=True),
        LSTM(32, return_sequences=False),
        RepeatVector(seq_len),
        LSTM(32, return_sequences=True),
        LSTM(64, return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == "__main__":
    # Test skeleton with dummy data
    import numpy as np
    model = build_lstm_autoencoder()
    model.summary()
    
    dummy_input = np.random.rand(16, 24, 2)
    output = model.predict(dummy_input)
    assert output.shape == dummy_input.shape
    print("Model skeleton forward pass successful.")
