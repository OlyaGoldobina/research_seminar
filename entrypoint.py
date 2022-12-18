import sys
from pathlib import Path

from connector.pg_connector import extract_data
from model.models_training import split_dataset, train_random_forest_model, train_neural_networks_model
from conf.conf import logging, settings
from sklearn.model_selection import train_test_split
from model.prediction import prediction


import argparse

parser = argparse.ArgumentParser(description='Predict the target on model and values')
parser.add_argument('--prediction_model',action='store', dest='prediction_model', type=str, nargs=1, help='model on which the target will predicts')
parser.add_argument('--prediction_params', action='append', dest='prediction_params', type=float, nargs=13, help='the values on target will predicts')
args = parser.parse_args()
model = args.prediction_model[0]
params = args.prediction_params

if model not in settings.MODELS:
    raise Exception("No such model, choose neural_networks or random_forest")
else:
    try:
        if model == "random_forest":
            predict = prediction(dir=settings.MODELS.random_forest, values=args.prediction_params)
            logging.info(f"Prediction for {model} is {predict}")
        elif model == "neural_networks":
            predict = prediction(dir=settings.MODELS.neural_networks, values=args.prediction_params)
            logging.info(f"Prediction for {model} is {predict}")
    except:
        df = extract_data(settings.DATA.link)
        x = df.iloc[:, :-1]
        y = df[settings.TARGET.target]

        X_train, X_test, y_train, y_test = split_dataset(x, y)

        if model == "random_forest":
            train_random_forest_model(X_train, y_train)
            predict = prediction(dir=settings.MODELS.random_forest, values=args.prediction_params)
            logging.info(f"Prediction for {model} is {predict}")
        elif model == "neural_networks":
            train_neural_networks_model(X_train, y_train)
            predict = prediction(dir=settings.MODELS.neural_networks, values=args.prediction_params)
            logging.info(f"Prediction for {model} is {predict}") 