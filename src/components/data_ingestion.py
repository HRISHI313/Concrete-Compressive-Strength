import sys
import os

from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataingestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','raw_data.csv')


class Dataingestion:
    def __init__(self):
        self.Ingestion_config = DataingestionConfig()

    def Initaite_Data_ingestion(self):
        logging.info('Data ingestion process started')
        try:
            df = pd.read_csv('Cement_Price_Prediction.csv')
            logging.info('Dataset has been read as a DataFrame')

            if not os.path.exists('artifacts'):
                os.makedirs('artifacts')

            df.to_csv(self.Ingestion_config.raw_data_path, header=True, index=False)
            logging.info('Raw data has been stored in the artifacts folder')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.Ingestion_config.train_data_path, header=True, index=False)
            test_set.to_csv(self.Ingestion_config.test_data_path, header=True, index=False)

            logging.info("Ingestion of the Data is completed")

            return self.Ingestion_config.train_data_path, self.Ingestion_config.test_data_path

        except Exception as e:
            logging.exception('Error occurred in Data ingestion Config: %s', e)
