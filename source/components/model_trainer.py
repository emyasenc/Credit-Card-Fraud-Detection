import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from source.exception import CustomException
from source.logger import logging
from source.utils import save_object, evaluate_models

current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(source_dir)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def preprocess_data(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)
        return X_train_normalized, X_test_normalized

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Starting model training")
            logging.info(f"Preprocessor path: {preprocessor_path}")
            
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[0],
                train_array[1],
                test_array[0],
                test_array[1]
            )
            logging.info("Data split into train and test sets")
            
            X_train, X_test = self.preprocess_data(X_train, X_test)
            
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier()
            }
            
            params = {
                "Random Forest": {
                    "n_estimators": [100, 200, 300, 400, 500],
                    "max_depth": [None, 10, 20, 30, 40],
                    "min_samples_split": [2, 5, 10, 15, 20],
                    "min_samples_leaf": [1, 2, 4, 6, 8]
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20, 30, 40],
                    "min_samples_split": [2, 5, 10, 15, 20],
                    "min_samples_leaf": [1, 2, 4, 6, 8]
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300, 400, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                    "max_depth": [3, 5, 7, 9, 11]
                },
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "max_iter": [100, 200, 300, 400, 500]
                },
                "K-Neighbors Classifier": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["ball_tree", "kd_tree", "brute"]
                },
                "XGBClassifier": {
                    "n_estimators": [100, 200, 300, 400, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                    "max_depth": [3, 5, 7, 9, 11]
                },
                "CatBoosting Classifier": {
                    "iterations": [100, 200, 300, 400, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                    "depth": [3, 5, 7, 9, 11]
                },
                "AdaBoost Classifier": {
                    "n_estimators": [50, 100, 200, 300, 400],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3]
                }
            }
            
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            # Get the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with accuracy: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy
        
        except Exception as e:
            raise CustomException(e, sys)