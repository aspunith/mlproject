import os
import sys
from src.exception import CustomException
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        
    except Exception as e:
        raise e
    

def evaluate_model(X_train,y_train,X_test,y_test, models,params):
    try:
        report={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = list(params.values())[i]
            
            if param is None or not isinstance(param, dict):
                raise ValueError(f"Parameter grid for model {list(models.keys())[i]} is not a dict or is None")
            
            grid = GridSearchCV(model, param, cv=5, n_jobs=-1)
            
            # Ensure `best_params` is always defined
            best_params = None

            try:
                grid.fit(X_train, y_train)
                best_params = grid.best_params_
            except Exception as e:
                print("Error during grid search:", str(e))
                print("Model:", model)
                print("Parameters:", params)

            if best_params is not None:
                model.set_params(**best_params)
            else:
                raise ValueError("The variable 'best_params' could not be determined due to an error during grid search.")
            
            model.fit(X_train, y_train)
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            
            report[list(models.keys())[i]] = r2_test
            
        return report
    
    except Exception as e:
        raise CustomException(f"Error in model evaluation: {str(e)}",sys)
    
def load_object(file_path):
        try:
            with open(file_path, 'rb') as file:
                obj = dill.load(file)
                
            return obj
        
        except Exception as e:
            raise CustomException(f"Error in loading object: {str(e)}",sys)
