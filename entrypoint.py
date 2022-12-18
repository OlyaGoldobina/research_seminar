import sys
from pathlib import Path

from connector.pg_connector import extract_data
from model.models_training import split_dataset, train_random_forest_model, train_neural_networks_model
from conf.conf import logging, settings
from sklearn.model_selection import train_test_split
from model.prediction import prediction


df = extract_data(settings.DATA.link)
x = df.iloc[:, :-1]
y = df[settings.TARGET.target]

X_train, X_test, y_train, y_test = split_dataset(x, y)

train_random_forest_model(X_train, y_train)
logging.info(f"Prediction for random forest is {prediction(settings.MODELS.random_forest, X_test)}")

train_neural_networks_model(X_train, y_train)
logging.info(f"Prediction for neural ntwork is {prediction(settings.MODELS.neural_network, X_test)}")
