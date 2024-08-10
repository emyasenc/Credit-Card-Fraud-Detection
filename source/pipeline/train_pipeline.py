import os
import sys
from source.exception import CustomException
from source.logger import logging
from source.components.data_ingestion import DataIngestion
from source.components.data_transformation import DataTransformation
from source.components.model_trainer import ModelTrainer
from source.utils import evaluate_models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_pipeline():
    try:
        # Data Ingestion
        logging.info("Starting data ingestion process...")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Data Transformation
        logging.info("Starting data transformation process...")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        # Model Training
        logging.info("Starting model training process...")
        model_trainer = ModelTrainer()
        model_report = model_trainer.initiate_model_trainer(train_arr, test_arr)

        # Model Evaluation
        logging.info("Evaluating models...")
        for model_name, model_info in model_report.items():
            model = model_info['best_model']
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_test_pred),
                "precision": precision_score(y_test, y_test_pred),
                "recall": recall_score(y_test, y_test_pred),
                "f1_score": f1_score(y_test, y_test_pred),
                "roc_auc_score": roc_auc_score(y_test, y_test_pred)
            }

            logging.info(f"Metrics for {model_name}: {metrics}")

        return model_report

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    train_pipeline()