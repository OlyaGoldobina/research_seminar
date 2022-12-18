import sys
from pathlib import Path

import pickle
from util.util import model_loading
from conf.conf import logging

def prediction(dir: str, values):
    """getting model prediction"""
    clf = model_loading(dir)
    logging.info("Model prediction starts")
    return clf.predict(values)    
