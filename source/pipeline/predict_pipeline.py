import sys
import os
import pandas as pd
from source.exception import CustomException
from source.utils import load_object

# Define numerical columns
numerical_columns = [
    "lat", "long", "city_pop", "amt", "distance_to_merchant", "age"
]

# Define categorical columns
categorical_columns = [
    "gender", "city", "state", "zip", "category"
]

# Define all columns
all_columns = [
    "cc_num", "gender", "city", "state", "zip", "lat", "long", "city_pop", "job", 
    "unix_time", "category", "amt", "is_fraud", "merchant", "merch_lat", "merch_long",
    "trans_year", "trans_month", "trans_day", "trans_hour", "trans_minute", "trans_second", 
    "day_of_week", "distance_to_merchant", "age"
]

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifact", "model.pkl")
            preprocessor_path = os.path.join("artifact", "preprocessor.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, cc_num: str, gender: str, city: str, state: str, zip: str, lat: float, long: float,
                 city_pop: int, job: str, unix_time: int, category: str, amt: float, is_fraud: int, 
                 merchant: str, merch_lat: float, merch_long: float, trans_year: int, trans_month: int,
                 trans_day: int, trans_hour: int, trans_minute: int, trans_second: int, day_of_week: int,
                 distance_to_merchant: float, age: int, **kwargs):
        self.cc_num = cc_num
        self.gender = gender
        self.city = city
        self.state = state
        self.zip = zip
        self.lat = lat
        self.long = long
        self.city_pop = city_pop
        self.job = job
        self.unix_time = unix_time
        self.category = category
        self.amt = amt
        self.is_fraud = is_fraud
        self.merchant = merchant
        self.merch_lat = merch_lat
        self.merch_long = merch_long
        self.trans_year = trans_year
        self.trans_month = trans_month
        self.trans_day = trans_day
        self.trans_hour = trans_hour
        self.trans_minute = trans_minute
        self.trans_second = trans_second
        self.day_of_week = day_of_week
        self.distance_to_merchant = distance_to_merchant
        self.age = age

    def get_data_as_data_frame(self):
        try:
            # Build dictionary for all features
            custom_data_input_dict = {
                "cc_num": [self.cc_num],
                "gender": [self.gender],
                "city": [self.city],
                "state": [self.state],
                "zip": [self.zip],
                "lat": [self.lat],
                "long": [self.long],
                "city_pop": [self.city_pop],
                "job": [self.job],
                "unix_time": [self.unix_time],
                "category": [self.category],
                "amt": [self.amt],
                "is_fraud": [self.is_fraud],
                "merchant": [self.merchant],
                "merch_lat": [self.merch_lat],
                "merch_long": [self.merch_long],
                "trans_year": [self.trans_year],
                "trans_month": [self.trans_month],
                "trans_day": [self.trans_day],
                "trans_hour": [self.trans_hour],
                "trans_minute": [self.trans_minute],
                "trans_second": [self.trans_second],
                "day_of_week": [self.day_of_week],
                "distance_to_merchant": [self.distance_to_merchant],
                "age": [self.age]
            }

            # Convert to DataFrame
            return pd.DataFrame(custom_data_input_dict, columns=all_columns)
        
        except Exception as e:
            raise CustomException(e, sys)