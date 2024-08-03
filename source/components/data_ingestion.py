## Read the dataset from a source, and then split the data into traininig and testing data
## NOTE: First set PYTHONPATH in terminal: export PYTHONPATH=/Users/sophia/Desktop/ml_projects
## Then execute: python /Users/sophia/Desktop/ml_projects/source/components/model_trainer.py
## Run in terminal by the command: python -m source.components.data_ingestion
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from source.exception import CustomException
from source.logger import logging
from source.components.data_transformation import DataTransformation
from source.components.data_transformation import DataTransformationConfig
from source.components.model_trainer import ModelTrainerConfig
from source.components.model_trainer import ModelTrainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact', "train2.csv")
    test_data_path: str = os.path.join('artifact', "test2.csv")
    raw_data_path: str = os.path.join('artifact', "combined_transactions.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Load the data with | delimiter
            df = pd.read_csv('/Users/sophia/Desktop/ml_projects/notebook/combined_transactions.csv', delimiter='|')
            logging.info('Read the dataset as dataframe')
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Save the raw data with | delimiter
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True, sep='|')
            
            logging.info("Train-test split initiated")
            
            # Identify the target column for stratification
            target_column = 'target'  # Update this to the actual target column name
            
            # Train-test split with stratified sampling
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df["is_fraud"])
            
            # Save the train and test data with , delimiter
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True, sep=',')
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True, sep=',')
            
            logging.info("Ingestion of the data is complete")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data, obj.ingestion_config.raw_data_path)
    
    # Separate features and targets
    X_train, y_train = train_arr
    X_test, y_test = test_arr
    
    # Traditional model training
    model_trainer = ModelTrainer()
    test_accuracy = model_trainer.initiate_model_trainer((X_train, y_train), (X_test, y_test), preprocessor_path)

