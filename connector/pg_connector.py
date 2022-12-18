import pandas as pd
import logging


def extract_data(link: str) -> pd.DataFrame:
    """ this function extracts csv data from the external source"""

    logging.info("Extracting data...")
    
    df = pd.read_csv(link)

    logging.info("The data is extracted")
    return df