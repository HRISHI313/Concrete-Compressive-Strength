import numpy as np
import pandas as pd

import pickle
import dill
import os
import sys

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import mean_squared_error, r2_score

def save_object (file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evalute_model(x_train,y_train,x_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(x_train,y_train)

            y_test_pred = model.predict(x_test)

            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            return report

    except Exception as e:
            logging.exception('Error occurred in Data ingestion Config: %s', e)