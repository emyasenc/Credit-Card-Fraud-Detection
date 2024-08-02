import logging
import os
import sys
import dill
from sklearn.metrics import accuracy_score
from source.exception import CustomException
from sklearn.model_selection import GridSearchCV

from joblib import Parallel, delayed
import time

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    def train_and_evaluate(model_name, model, param_grid, X_train, y_train, X_test, y_test):
        start_time = time.time()
        logging.info(f"Training {model_name} started")

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        train_model_score = accuracy_score(y_train, y_train_pred)
        test_model_score = accuracy_score(y_test, y_test_pred)
        
        end_time = time.time()
        logging.info(f"Training {model_name} completed in {end_time - start_time:.2f} seconds")
        logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        
        return model_name, test_model_score, best_model, grid_search.best_params_

    try:
        results = Parallel(n_jobs=-1)(
            delayed(train_and_evaluate)(model_name, model, params[model_name], X_train, y_train, X_test, y_test)
            for model_name, model in models.items()
        )
        
        report = {model_name: test_model_score for model_name, test_model_score, _, _ in results}
        best_model_info = max(results, key=lambda item: item[1])
        
        best_model_name = best_model_info[0]
        best_model_score = best_model_info[1]
        best_model = best_model_info[2]
        best_params = best_model_info[3]
        
        logging.info(f"Best model: {best_model_name} with score {best_model_score} and parameters {best_params}")
        
        return report, best_model, best_params
    
    except Exception as e:
        logging.error("Error during model evaluation", exc_info=True)
        raise CustomException(e, sys)