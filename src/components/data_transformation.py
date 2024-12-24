#This file contains code for data transformation like
# - Data cleaning, filling missing values, encoding categorical variables, etc.
# - Feature engineering, creating new features, etc.
# - Data transformation, scaling, normalization, etc.

import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer #ColumnTransformer is used to apply different transformations to different columns, used in pipelines
from sklearn.impute import SimpleImputer #SimpleImputer is used to fill the missing values in the dataset
from sklearn.pipeline import Pipeline #Pipeline is used to chain multiple estimators into one
from sklearn.preprocessing import OneHotEncoder, StandardScaler #OneHotEncoder is used to encode the categorical variables, StandardScaler is used to scale the features

from src.exception import CustomException
from src.utils import save_object
from src.logger import logging

import os


@dataclass #dataclass decorator is used to create a class with attributes and methods
class DataTransfromationConfig:
    preprocessor_obj_file_path: str=os.path.join('artifacts','preprocessor.pkl') #save the preprocessor object in the artifacts folder

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransfromationConfig()
        
    def get_data_transformer_object(self):
        
        """
        This Function is responsible for data transformation
        """
        
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            
            #Create a column transformer object, which will apply different transformations to different columns
            num_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')), #fill the missing values with the median
                ('scaler', StandardScaler(with_mean=False)) #scale the features
            ]
                )
            
            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), #fill the missing values with the most frequent value
                ('one_hot_encoder', OneHotEncoder()), #encode the categorical variables
                ('scaler', StandardScaler(with_mean=False)) #scale the features
            ]
                )
            
            logging.info("Numerical columns standard scaling completed")
            
            logging.info("Catgorical columns one hot encoding completed")
            
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns), #apply the num_pipeline to numerical columns
                    ('cat_pipeline', cat_pipeline, categorical_columns) #apply the cat_pipeline to categorical columns
                ]
            )

            return preprocessor
        

        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train data completed")
            
            logging.info("Obtaining Preprocessor object")
            
            preprocessing_obj=self.get_data_transformer_object()
            target_column_name = 'math_score'
            numerical_columns = ['writing_score','reading_score']
            
            input_feature_train_df = train_df.drop(target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] #np.c_ is used to concatenate the arrays column-wise
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Saved Preprocessing object")
            
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr
                #self.transformation_config.preprocessor_obj_file_path
            )
            
            
            
        except Exception as e:
            raise CustomException(e,sys)
        
        