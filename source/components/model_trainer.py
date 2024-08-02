import os
import sys
from dataclasses import dataclass
import time

from catboost import CatBoostClassifier
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

from source.exception import CustomException
from source.logger import logging

from source.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Starting model training")
            logging.info(f"Preprocessor path: {preprocessor_path}")
            
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info("Data split into train and test sets")
            
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
                    "penalty": ["l1", "l2", "elasticnet", "none"],
                    "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    "solver": ["lbfgs", "liblinear", "saga", "newton-cg"]
                },
                "K-Neighbors Classifier": {
                    "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
                },
                "XGBClassifier": {
                    "n_estimators": [100, 200, 300, 400, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                    "max_depth": [3, 5, 7, 9, 11],
                    "colsample_bytree": [0.3, 0.5, 0.7, 0.9, 1.0],
                    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0]
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
            
            logging.info("Models and hyperparameters defined")
            start_time = time.time()
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            logging.info(f"Model evaluation report: {model_report}")
            logging.info(f"Total model evaluation time: {time.time() - start_time:.2f} seconds")
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            logging.info(f"Best model score: {best_model_score}")
            
            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name}")

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found with sufficient accuracy")
            
            # Ensure the best model is fitted properly before saving and using it
            best_model.fit(X_train, y_train)
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Model saved successfully")
            
            predicted = best_model.predict(X_test)
            acc_score = accuracy_score(y_test, predicted)
            logging.info(f"Accuracy score: {acc_score}")
            return acc_score
            
        except Exception as e:
            logging.error("Error in model training", exc_info=True)
            raise CustomException(e, sys)