from model_pipeline import preprocess_device

def create_clients(df, features):
    clients = {}

    for device in df['device'].unique():
        d = df[df['device'] == device].copy()

        if len(d) < 1000:
            continue

        X_train, X_test, scaler = preprocess_device(d, features)

        clients[device] = {
            "X_train": X_train,
            "X_test": X_test,
            "scaler": scaler
        }

    return clients