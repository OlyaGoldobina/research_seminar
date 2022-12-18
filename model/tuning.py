import sys
from pathlib import Path

from conf.conf import logging, settings
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

def model_tuning(model, X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict):
    """tuning model"""
    logging.info(f"Strarting GridSearch for {model}")

    searcher = GridSearchCV(model, params, scoring="neg_root_mean_squared_error", cv=settings.PARAMS.cv) #give gyperparameters to our GridSearch
    searcher.fit(X_train, y_train) #fit GridSearch
    best_model = searcher.best_estimator_

    logging.info("GridSearch is finished")
    return best_model