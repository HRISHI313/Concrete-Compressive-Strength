import os
import sys

PROJECT_ROOT = r"D:\pythonProject\Cement Price Prediction"
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

from src.components.data_ingestion import Dataingestion
from src.components.data_transformation import DataTransformation

if __name__ == '__main__':
    # Create an instance of the DataIngestion class
    dataingestion = Dataingestion()

    # Call the method to initiate data ingestion
    train_data_path,test_data_path = dataingestion.Initaite_Data_ingestion()

    # Creating an instance of the Data Transformation class
    datatransformation = DataTransformation()

    #Call the method to initiate data transformation
    train_arr,test_arr,_ = datatransformation.initiate_data_transformation(train_data_path,test_data_path)
