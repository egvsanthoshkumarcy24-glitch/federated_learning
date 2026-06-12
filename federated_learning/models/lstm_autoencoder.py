from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

def build_lstm_autoencoder(seq_len, n_features):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(seq_len, n_features), return_sequences=True),
        LSTM(32, activation='relu', return_sequences=False),
        RepeatVector(seq_len),
        LSTM(32, activation='relu', return_sequences=True),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
