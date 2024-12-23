#Contains the code for data ingestion: usually reading the data from the source and loading it into the memory for further processing.Like from csv, database, etc.
import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv') #save the train data in the artifacts folder
    test_data_path: str=os.path.join('artifacts','test.csv') #save the test data in the artifacts folder
    raw_data_path: str=os.path.join('artifacts','data.csv') #save the raw data in the artifacts folder
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        
        try:
            df =pd.read_csv('notebook/data/stud.csv')
            logging.info("Data read successfully as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) #create the directory if it does not exist
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) #save the raw data in the artifacts folder
            
            logging.info("Train Test Split Initiated")
            train_set, test_set=train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) #save the train data in the artifacts folder
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True) #save the test data in the artifacts folder
            
            logging.info("Ingestion of the data is completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
            
if __name__ == "__main__":
    data_ingestion=DataIngestion()
    data_ingestion.initiate_data_ingestion()
