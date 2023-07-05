import sys
import os

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationconfig:
    preprocessor_ob_file_path: str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        logging.info('Data transformation has been initiated')
        try:
            numerical_col = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer',
                             'Coarse Aggregate', 'Fine Aggregate', 'Age']

            logging.info('Numerical pipeline is initiated')
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scalar', StandardScaler())
                ]
            )

            logging.info('Column Transformation initiated')
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_col)
            ])

            return preprocessor

        except CustomException as e:
            logging.error('Error Occurred in get_data_transformation_object: %s', e)

    def initiate_data_transformation(self, train_path, test_path):
        logging.info('initiate_data_transformation has been initiated')
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Preprocessing object is getting created')
            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Concrete compressive strength'
            drop_column_name = 'Concrete compressive strength'

            input_feature_train_df = train_df.drop(columns=drop_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training dataframe and testing dataframe')

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle is created and saved')
            logging.info('Data Transformation has been completed')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )

        except Exception as e:
            logging.exception('Error occurred in initiate_data_transformation: %s', e)
