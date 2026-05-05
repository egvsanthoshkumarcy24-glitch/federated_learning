import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

SEQ_LEN = 24

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        LSTM(32, return_sequences=False),
        RepeatVector(input_shape[0]),
        LSTM(32, return_sequences=True),
        LSTM(64, return_sequences=True),
        TimeDistributed(Dense(input_shape[1]))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_sequences(data, seq_len=SEQ_LEN):
    return np.array([data[i:i+seq_len] for i in range(len(data)-seq_len)])

def preprocess_device(df, features):
    scaler = MinMaxScaler()

    split = int(0.7 * len(df))
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()

    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features]  = scaler.transform(test_df[features])

    X_train = create_sequences(train_df[features].values)
    X_test  = create_sequences(test_df[features].values)

    return X_train, X_test, scaler

def train_local(X_train, epochs=10, batch_size=64):
    model = build_model((SEQ_LEN, X_train.shape[2]))

    es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[es],
        verbose=0
    )

    return model

def get_weights(model):
    return model.get_weights()

def set_weights(model, weights):
    model.set_weights(weights)