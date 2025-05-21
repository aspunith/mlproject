#This file contains model training code
import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

# Ensure the library is installed in the environment
try:
    import catboost
except ImportError:
    raise ImportError("The 'catboost' library is not installed. Please install it using 'pip install catboost'.")

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
from sklearn.base import BaseEstimator, RegressorMixin

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
import numpy as np

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str =os.path.join('artifacts', 'trained_model.pkl')


from sklearn.base import BaseEstimator, RegressorMixin

class XGBRegressorWrapper(XGBRegressor, BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __sklearn_tags__(self):
        return {
            'non_deterministic': True,
            'requires_positive_X': False,
            'requires_positive_y': False,
            'X_types': ['2darray'],
            'poor_score': False,
            'no_validation': False,
            'multioutput': False,
            'multioutput_only': False,
            'allow_nan': False,
            'stateless': False,
            'binary_only': False,
            'requires_fit': True,
            'preserves_dtype': [np.float64],
            'requires_y': True,
            'pairwise': False,
        }
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
                "XGBoost": XGBRegressorWrapper(),
                "CatBoost": CatBoostRegressor(),
                "KNN": KNeighborsRegressor(),
                "Linear Regression": LinearRegression(),
                
            }
            params = {
        "Random Forest": {
            "n_estimators": [100, 200, 300],    # Number of trees in the forest
            "max_depth": [None, 10, 20],        # Maximum depth of the tree
            "min_samples_split": [2, 5, 10]     # Minimum samples required to split a node
    },
        "Decision Tree": {
            "max_depth": [None, 10, 20],        # Maximum depth of the tree
            "min_samples_split": [2, 5, 10],    # Minimum samples required to split a node
            "min_samples_leaf": [1, 2, 4]       # Minimum samples at a leaf node
    },
        "AdaBoost": {
            "n_estimators": [50, 100, 200],     # Number of boosting stages
            "learning_rate": [0.01, 0.1, 1.0],  # Shrinks the contribution of each regressor
            "loss": ['linear', 'square', 'exponential']  # Loss function
    },
        "Gradient Boosting": {
            "n_estimators": [100, 200, 300],    # Number of boosting stages
            "learning_rate": [0.01, 0.1, 0.2],  # Step size shrinkage
            "max_depth": [3, 5, 7]              # Maximum depth of the tree
    },
        "XGBoost": {
            "n_estimators": [100, 200, 300],    # Number of boosting rounds
            "learning_rate": [0.01, 0.1, 0.2],  # Step size shrinkage
            "max_depth": [3, 5, 7]              # Maximum depth of a tree
    },
        "CatBoost": {
            "iterations": [500, 1000, 1500],    # Number of boosting iterations
            "learning_rate": [0.01, 0.03, 0.1], # Learning rate
            "depth": [4, 6, 8]                  # Depth of the tree
    },
        "KNN": {
            "n_neighbors": [3, 5, 7],           # Number of neighbors to use
            "weights": ['uniform', 'distance'], # Weight function
            "p": [1, 2, 3]                      # Power parameter for Minkowski metric
    },
        "Linear Regression": {
            "fit_intercept": [True, False],     # Whether to calculate the intercept
            #"normalize": [True, False],         # Whether to normalize the data
            "n_jobs": [None, 1, -1]             # Number of jobs to use for computation
    }
}

            
            model_report:dict=evaluate_model(X_train, y_train,X_test,y_test, models=models,params=params)
            
            #best model score
            best_model_score = max(model_report.values())
            
            #best model name
            best_model_name = [key for key in model_report if model_report[key]==best_model_score][0]
            best_model = models[best_model_name]
            
            #threshold for best model
            if best_model_score<0.6:
                raise CustomException('Best model score is less than 0.6, no model found', sys.exc_info())
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

