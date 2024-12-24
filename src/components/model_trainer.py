#This file contains model training code
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str =os.path.join('artifacts', 'trained_model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Splitting training and test input data')
            X_train,y_train, X_test, y_test =(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(),
                "KNN": KNeighborsRegressor(),
                "Linear Regression": LinearRegression(),
                
            }
            
            model_report:dict=evaluate_model(X_train, y_train,X_test,y_test, models=models)
            
            #best model score
            best_model_score = max(model_report.values())
            
            #best model name
            best_model_name = [key for key in model_report if model_report[key]==best_model_score][0]
            best_model = models[best_model_name]
            
            #threshold for best model
            if best_model_score<0.6:
                raise CustomException('Best model score is less than 0.6, no model found')
            logging.info(f'Best model found on both training and testing dataset: {best_model_name}')
            
            save_object(
                self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            r2=r2_score(y_test, predicted)
            logging.info(f'R2 Score of the best model is {r2}')
            
            return r2
            
        except Exception as e:
            raise CustomException(e,sys)
        
