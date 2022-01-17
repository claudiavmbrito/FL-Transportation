import warnings
import flwr as fl
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

import utils

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = utils.get_dataset()

    # Split train set into N partitions and randomly use one for training.
    partition_id = np.random.choice(10)
    (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

    model = RandomForestClassifier(n_estimators = 15)

    utils.set_initial_params(model)

    # Define Flower client
    class TMCClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['rnd']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8080", client=TMCClient())