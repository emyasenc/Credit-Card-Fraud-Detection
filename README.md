# Credit Card Fraud Detection Model

## Description
The Credit Card Fraud Detection Model is designed to identify fraudulent transactions within a credit card dataset. By leveraging traditional machine learning algorithms, the model classifies transactions as either legitimate or fraudulent based on anonymized features. This solution enhances security by automatically flagging suspicious activities, reducing the risk of financial loss, and improving overall fraud detection efficiency.

## Implementation
The implementation of this model involves the following key components:

### Data Ingestion and Preparation
- **Data Files**: The model utilizes two primary datasets: `train2.csv` for training and `test2.csv` for evaluation. These files contain anonymized transaction features and labels indicating whether a transaction is fraudulent.
- **Stratified Sampling**: Stratified sampling is employed to ensure representative distribution across different classes, allowing for efficient handling of large datasets.

### Data Processing
- **Preprocessing**: Data is normalized using `StandardScaler` to bring features to a common scale, improving the performance and convergence speed of the machine learning algorithms.
- **Feature Engineering**: Relevant features are extracted and transformed to prepare the data for model training.

### Model Training
- **Algorithms**: The following traditional machine learning algorithms are implemented:
  - Random Forest
  - Decision Tree
  - Gradient Boosting
  - Logistic Regression
  - K-Neighbors Classifier
  - XGBoost Classifier
  - CatBoost Classifier
  - AdaBoost Classifier
- **Hyperparameter Tuning**: Grid search is utilized to optimize hyperparameters for each model, ensuring the best possible performance.

### Model Evaluation
- **Metrics**: The model is evaluated based on accuracy, using metrics provided by Scikit-Learn. Performance is assessed on the test dataset to determine the effectiveness of each algorithm.
- **Model Selection**: The model with the highest accuracy score is selected as the final model.

### Saving the Model
- **Persistence**: The trained models are saved to disk using `dill`, allowing for easy loading and usage for future predictions without retraining.

## How to Use
1. **Setup**: Ensure all dependencies are installed, including Scikit-Learn and other required libraries.
2. **Data Preparation**: Place your data files (`train2.csv` and `test2.csv`) in the appropriate directory.
3. **Training**: Run the `model_trainer.py` script to train the model. This script performs data preprocessing, trains multiple models, and saves the best models for your selection.
4. **Prediction**: Use the trained model to make predictions on new transaction data by loading the model and applying it to the dataset.
5. **Evaluation**: Check the accuracy of the model using the provided evaluation metrics.

## Dependencies
- Python 3.x
- Scikit-Learn
- Dill
- Joblib
- XGBoost
- CatBoost

## Main Files
- **`model_trainer.py`**: Script for training and evaluating models.
- **`data_ingestion.py`**: Script for loading and preprocessing data.
- **`data_transformation.py`**: Script for feature engineering and transformation.
- **`utils.py`**: Utility functions for saving models and evaluating performance.
- **`predict_pipeline.py`**: Script for making predictions with the trained model.
- **`train_pipeline.py`**: Script for managing the end-to-end training pipeline.

## License
This project is licensed under the MIT License.
