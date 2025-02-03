from src.exception import CustomException
from src.logs import logging
from src.components.model_trainer import ModelTrainerConfig
from src.utils.utils import input_preprocess_image

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

class PredictPipeline:
    def __init__(self):
        self.model_path = ModelTrainerConfig()

    def predict(self,img_path):
        try:
            loaded_model = tf.keras.models.load_model(self.model_path.trained_model_file_path)
            img_array = input_preprocess_image(img_path)  # Ensure img_path is passed here
        
            # Perform the prediction
            predictions = loaded_model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=-1)
            
            return predicted_class

        except Exception as e:
            raise CustomException(e,sys)