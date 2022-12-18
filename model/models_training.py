import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import sys
sys.path.append('.')

from .tuning import model_tuning
from util.util import model_dumping
import sys
from pathlib import Path

from conf.conf import logging, settings
from connector.pg_connector import extract_data


def split_dataset(x: pd.DataFrame, y: pd.DataFrame) -> list:
    """split variable into train and test datasets"""

    logging.info("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=3)
    
    logging.info("Dataset is split")

    return X_train, X_test, y_train, y_test


def train_random_forest_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    """initializing and training the model"""

    logging.info("Training random forest model...")

    clf = model_tuning(model=RandomForestClassifier(),
                                              X_train=X_train,
                                              y_train=y_train,
                                              params=settings.TUNING.RandomForest)

    logging.info("The random forest model is trained")

    model_dumping(dir=settings.MODELS.random_forest, model=clf)


def train_neural_networks_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    """initializing and training the model"""

    logging.info("Training nueral networks model...")

    clf = model_tuning(model=MLPClassifier(),
                                              X_train=X_train,
                                              y_train=y_train,
                                              params=settings.TUNING.NeuralNetworks)
    clf.fit(X_train, y_train)

    logging.info("The neural network model is trained")

    model_dumping(dir=settings.MODELS.neural_networks, model=clf)
