import os
import sys

PROJECT_ROOT = r"D:\pythonProject\Cement Price Prediction"
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

from src.components.data_ingestion import Dataingestion

if __name__ == '__main__':
    # Create an instance of the DataIngestion class
    obj = Dataingestion()

    # Call the method to initiate data ingestion
    train_data_path,test_data_path = obj.Initaite_Data_ingestion()


