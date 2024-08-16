# Credit Card Fraud Detection Model

## Description
### The Credit Card Fraud Detection Model is designed to identify fraudulent transactions in a credit card dataset. The model leverages traditional machine learning algorithms to classify transactions as either legitimate or fraudulent based on anonymized features provided in the dataset. This solution is aimed at enhancing security by automatically flagging suspicious activities, reducing the risk of financial loss, and improving overall fraud detection efficiency.

## Implementation
### The implementation of this model involves the following key components:

## Data Ingestion and Preparation

### * Data Files: The model utilizes two primary datasets: train2.csv for training and test2.csv for evaluation. These files contain anonymized transaction features and labels indicating whether a transaction is fraudulent or not.
### * Stratified Sampling: To handle large datasets efficiently, stratified sampling is employed to ensure representative distribution across different classes.

## Data Processing

### Preprocessing: Data is normalized using StandardScaler to bring features to a common scale, improving the performance and convergence speed of the machine learning algorithms.

### Feature Engineering: Relevant features are extracted and transformed to prepare the data for model training.

## Model Training

 ### * Algorithms: Several traditional machine learning algorithms are implemented, including:
 ### * Random Forest
 ### * Decision Tree
 ### * Gradient Boosting
 ### * Logistic Regression
 ### * K-Neighbors Classifier
 ### * XGBoost Classifier
 ### * CatBoost Classifier
 ### * AdaBoost Classifier
 
 ### Hyperparameter Tuning: Grid search is used to optimize hyperparameters for each model, ensuring the best possible performance.

## Model Evaluation

### Metrics: The model is evaluated based on accuracy, using metrics provided by Scikit-Learn. Performance is assessed on the test dataset to determine the effectiveness of each algorithm.

### Model Selection: The model with the highest accuracy score is selected as the final model.
Saving the Model

### Persistence: The trained models are saved to disk using dill, ensuring that it can be easily loaded and used for future predictions without retraining.

## How to Use
  ### * Setup: Ensure all dependencies are installed, including Scikit-Learn, PySpark, and other required libraries.
  ### * Data Preparation: Place your data files (train2.csv and test2.csv) in the appropriate directory.
  ### * Training: Run the model_trainer.py script to train the model. The script will perform data preprocessing, train multiple models, and save the best-performing model.
  ### * Prediction: Use the trained model to make predictions on new transaction data by loading the model and applying it to the dataset.
  ### * Evaluation: Check the accuracy of the model using the provided evaluation metrics.

## Dependencies
  ### * Python 3.x
  ### * Scikit-Learn
  ### * PySpark
  ### * Dill
  ### * Joblib
  ### * XGBoost
  ### * CatBoost

## Main Files
  ### * model_trainer.py: Script for training and evaluating models.
  ### * data_ingestion.py: Script for loading and preprocessing data.
  ### * data_transformation.py: Script for feature engineering and transformation.
  ### * utils.py: Utility functions for saving models and evaluating performance.
  ### * predict_pipeline.py: Script for making predictions with the trained model.
  ### * train_pipeline.py: Script for managing the end-to-end training pipeline.

## License
  ### This project is licensed under the MIT License.
