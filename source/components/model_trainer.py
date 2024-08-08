import os
import sys
import time
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from source.exception import CustomException
from source.logger import logging
from source.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            params = {
                "Random Forest": {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 10],
                    'min_samples_leaf': [1, 5]
                },
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 10],
                    'min_samples_leaf': [1, 5]
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5]
                },
                "Logistic Regression": {
                    'C': [0.01, 1, 100],
                    'max_iter': [100, 300]
                },
                "K-Neighbors Classifier": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['ball_tree', 'kd_tree']
                },
                "XGBClassifier": {
                    'n_estimators': [100, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5]
                },
                "CatBoosting Classifier": {
                    'iterations': [100, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [3, 5]
                },
                "AdaBoost Classifier": {
                    'n_estimators': [50, 200],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }

            model_report = {}

            for model_name, model in models.items():
                logging.info(f"Training {model_name} with parameters: {params[model_name]}")
                start_time = time.time()

                grid_search = RandomizedSearchCV(model, params[model_name], n_iter=5, scoring='accuracy', n_jobs=-1, cv=5, verbose=1)
                grid_search.fit(X_train, y_train)

                end_time = time.time()
                elapsed_time = end_time - start_time
                logging.info(f"Training {model_name} took {elapsed_time:.2f} seconds")

                best_model = grid_search.best_estimator_
                best_score = grid_search.best_score_
                logging.info(f"{model_name} best score: {best_score:.4f}")

                model_report[model_name] = {
                    "best_model": best_model,
                    "best_score": best_score,
                    "training_time": elapsed_time
                }

                # Save best model
                save_object(
                    file_path=os.path.join("artifact", f"{model_name}_best_model.pkl"),
                    obj=best_model
                )

            logging.info("Model training completed for all models")
            return model_report
        
        except Exception as e:
            raise CustomException(e, sys)