# Transportation Mode Classification using Flower and scikit-learn

This example of Flower uses `scikit-learn`'s `RandomForest` model to train a federated learning system and save it. 

## Project Setup

First clone the project. Following the single line code, we can clone and define the folder to run this simple approach. 

```shell
git clone --depth=1 https://github.com/claudiavmbrito/FL-Transportation.git && mv FL-Transportation/iot . && rm -rf FL-Transportation && cd iot
```

This will create a new directory called `iot` containing the following files:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- utils.py
-- README.md
```

### From Flower: 
Project dependencies (such as `scikit-learn` and `flwr`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
poetry run python3 -c "import flwr"
```

If you don't see any errors you're good to go!

# Run the project 

After the previous steps, the project can be run:

```shell
poetry run python3 server.py
```

Now you are ready to start the Flower clients which will participate in the learning. To do so simply open two more terminals and run the following command in each:

```shell
poetry run python3 client.py
```

Then, Flower starts the federated learning training. 