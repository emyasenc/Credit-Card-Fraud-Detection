## NOTE: Run in terminal by the command: python -m source.components.data_transformation
import sys
import os
import subprocess
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from source.exception import CustomException
from source.logger import logging
from source.utils import save_object

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifact', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_ob(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            # Define numerical columns
            numerical_columns = [
                "zip", "lat", "long", "city_pop", "amt", "merch_lat", "merch_long"
            ]
            
            # Define categorical columns
            categorical_columns = [
                "gender", "state", "category", "merchant"
            ]
            
            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown='ignore'))
                ]
            )
            
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path, combined_path):
        try:
            # Load data with appropriate delimiters
            combined_df = pd.read_csv(combined_path, delimiter='|')
            train_df = pd.read_csv(train_path, delimiter=',')
            test_df = pd.read_csv(test_path, delimiter=',')
        
            logging.info("Reading the combined, training, and testing data completed")
            logging.info("Obtaining preprocessing object")
        
            preprocessing_obj = self.get_data_transformer_ob()
        
            target_column_name = "is_fraud"
        
            # Separate input features and target feature
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
        
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
        
            logging.info('Applying preprocessing object on training dataframe and testing dataframe.')
        
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
        
            # Convert sparse matrices to dense arrays
            input_feature_train_arr = input_feature_train_arr.toarray()
            input_feature_test_arr = input_feature_test_arr.toarray()
        
            # No need to convert target features to categorical
            target_feature_train_arr = target_feature_train_df.values
            target_feature_test_arr = target_feature_test_df.values
        
            # Check the types and shapes of arrays
            logging.info(f"Type of input_feature_train_arr: {type(input_feature_train_arr)}")
            logging.info(f"Type of target_feature_train_arr: {type(target_feature_train_arr)}")
            logging.info(f"Type of input_feature_test_arr: {type(input_feature_test_arr)}")
            logging.info(f"Type of target_feature_test_arr: {type(target_feature_test_arr)}")
        
            # Check the shapes before concatenating
            logging.info(f"Train input features shape: {input_feature_train_arr.shape}")
            logging.info(f"Train target shape: {target_feature_train_arr.shape}")
            logging.info(f"Test input features shape: {input_feature_test_arr.shape}")
            logging.info(f"Test target shape: {target_feature_test_arr.shape}")
        
            # Concatenate input features and target features
            train_arr = (input_feature_train_arr, target_feature_train_arr)
            test_arr = (input_feature_test_arr, target_feature_test_arr)
        
            logging.info("Saved preprocessing object.")
        
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )
        
            # Call model_trainer.py after transformation
            self.run_model_trainer()

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


    def run_model_trainer(self):
        """Function to execute model_trainer.py"""
        try:
            # Log the start of model training
            logging.info("Starting model_trainer.py")
            
            # Log the current working directory
            logging.info(f"Current working directory: {os.getcwd()}")
            
            # Define the absolute path to model_trainer.py
            model_trainer_path = os.path.join(os.getcwd(), 'source/components/model_trainer.py')
            
            # Run the model_trainer script
            result = subprocess.run(["python", model_trainer_path], capture_output=True, text=True)
            
            # Log the output and errors from the script
            logging.info("model_trainer.py output:\n" + result.stdout)
            logging.error("model_trainer.py errors:\n" + result.stderr)
            
            if result.returncode != 0:
                raise Exception(f"model_trainer.py failed with return code {result.returncode}")

            logging.info("model_trainer.py executed successfully")

        except Exception as e:
            logging.error("Error running model_trainer.py", exc_info=True)
            sys.exit(1)

