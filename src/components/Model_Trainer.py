import sys
import os

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evalute_model

import pandas as pd
import numpy as np
from dataclasses import dataclass

#Model Training
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainConfig:
    train_model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.Model_trainer_config = ModelTrainConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Splitting Dependent and Independent features')
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                'CatBoostRegressor':CatBoostRegressor()
            }
            model_report:dict = evalute_model(x_train,y_train,x_test,y_test,models)
            print(model_report)
            print('\n===========================================================================================')
            logging.info(f'Model Report: {model_report}')

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f'Best Model Found , Model name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n============================================================================================')
            logging.info(f'Best Model found, Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                file_path=self.Model_trainer_config.train_model_path,
                obj = best_model
            )

        except Exception as e:
            logging.exception('Error occurred in Data ingestion Config: %s', e)
