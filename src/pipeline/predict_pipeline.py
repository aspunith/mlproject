import sys
import pandas as pd
from src.exception import CustomException
from src.components.data_transformation import DataTransformation, DataTransfromationConfig
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
           pass
    
    def predict(self,features):
        try:
            model_path = "artifacts/trained_model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(model_path)
            preprocessor_path = load_object(preprocessor_path)
            data_scaled = preprocessor_path.transform(features)
            preds = model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
class CustomData: #class useful for mapping the html data to the backend.
    def __init__(self,
                gender: str,
                race_ethnicity: str,
                parental_level_of_education: str,                
                lunch: str,
                test_preparation_course: str,
                reading_score: int,
                writing_score: int):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity if race_ethnicity is not None else "Unknown"
        self.parental_level_of_education = parental_level_of_education if parental_level_of_education is not None else "Unknown"
        self.lunch = lunch if lunch is not None else "Unknown"
        self.test_preparation_course = test_preparation_course if test_preparation_course is not None else "Unknown"
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            
            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            raise CustomException(e,sys)
        