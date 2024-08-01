import os
import sys

import numpy as np
import pandas as pd
import dill

from source.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_pathh = os.path.dirname(file_path)
        
        os.makedirs(dir_pathh, exist_ok= True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)