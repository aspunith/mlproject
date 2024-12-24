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
            params = list(params.values())[i]
            
            grid = GridSearchCV(model, params, cv=5, n_jobs=-1)
            grid.fit(X_train, y_train)
            
            model.set_params(**grid.best_params_)
            
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            
            report[list(models.keys())[i]] = r2_test
            
            
            
        return report
    
    except Exception as e:
        raise CustomException(f"Error in model evaluation: {str(e)}")
