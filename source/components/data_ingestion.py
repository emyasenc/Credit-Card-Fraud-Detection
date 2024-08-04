## Read the dataset from a source, and then split the data into traininig and testing data
## NOTE: First set PYTHONPATH in terminal: export PYTHONPATH=/Users/sophia/Desktop/ml_projects
## Then execute: python /Users/sophia/Desktop/ml_projects/source/components/model_trainer.py
## Run in terminal by the command: python -m source.components.data_ingestion
import os
import sys
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
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
    train_data_path: str = os.path.join('artifact', "train.csv")
    test_data_path: str = os.path.join('artifact', "test.csv")
    raw_data_path: str = os.path.join('artifact', "combined_transactions.csv")

class DataIngestion:
    def __init__(self, sample_size=10000):
        self.ingestion_config = DataIngestionConfig()
        self.sample_size = sample_size  # Add sample size attribute
        
    def stratified_sampling(self, data, target_column, sample_size, test_size=0.2, random_state=42):
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Initialize StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

        # Perform the split
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Control the sample size
        if sample_size < len(X_train):
            X_train = X_train.sample(n=sample_size, random_state=random_state)
            y_train = y_train.loc[X_train.index]

        return X_train, X_test, y_train, y_test

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Load the data with | delimiter
            df = pd.read_csv('/Users/sophia/Desktop/ml_projects/notebook/all_transactions.csv', delimiter='|')
            logging.info('Read the dataset as dataframe')
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Save the raw data with | delimiter
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True, sep='|')
            
            logging.info("Train-test split initiated")
            
            # Identify the target column for stratification
            target_column = 'is_fraud'  # Update this to the actual target column name
            
            # Perform stratified sampling
            X_train, X_test, y_train, y_test = self.stratified_sampling(df, target_column, self.sample_size)

            # Combine features and target for saving
            train_set = pd.concat([X_train, y_train], axis=1)
            test_set = pd.concat([X_test, y_test], axis=1)
            
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
    sample_size = 600
    obj = DataIngestion(sample_size=sample_size)
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data, obj.ingestion_config.raw_data_path)
    
    # Separate features and targets
    X_train, y_train = train_arr
    X_test, y_test = test_arr
    
    # Traditional model training
    model_trainer = ModelTrainer()
    test_accuracy = model_trainer.initiate_model_trainer((X_train, y_train), (X_test, y_test), preprocessor_path)