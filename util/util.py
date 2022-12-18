import pickle
import sys
from pathlib import Path

from conf.conf import logging

def model_dumping(model, dir: str) -> None:
    """dumping model into picke file"""

    logging.info(f"Dumping model to {dir}")
    pickle.dump(model, open(dir, 'wb'))
    logging.info(f"The model is saved")


def model_loading(dir: str):
    """loading model from picke file"""

    logging.info(f"Loading model from {dir}")
    pickled_model = pickle.load(open(dir, 'rb'))
    logging.info(f"The model is loaded")
    return pickled_model
