import flwr as fl
import tensorflow as tf
import numpy as np

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, X_val, y_val):
        self.model = model
        self.X_train = X_train
        self.X_val = X_val
        self.y_val = y_val

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        epochs = config.get("epochs", 1)
        batch_size = config.get("batch_size", 32)
        
        history = self.model.fit(
            self.X_train, self.X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val, self.X_val),
            verbose=0
        )
        
        results = {
            "loss": history.history["loss"][0],
            "val_loss": history.history["val_loss"][0],
        }
        return self.model.get_weights(), len(self.X_train), results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        
        # Calculate reconstruction MSE for validation set
        preds = self.model.predict(self.X_val, verbose=0)
        mse = np.mean(np.power(self.X_val - preds, 2), axis=(1, 2))
        
        # Determine a quick threshold (e.g. 95th percentile of validation MSE) to compute an F1
        # In actual practice, threshold should be calculated on train set.
        threshold = np.percentile(mse, 95)
        y_pred = (mse > threshold).astype(int)
        
        from sklearn.metrics import f1_score
        f1 = f1_score(self.y_val, y_pred, zero_division=0)
        
        # Loss is MSE of reconstruction
        loss = np.mean(mse)
        return loss, len(self.X_val), {"f1": f1}
