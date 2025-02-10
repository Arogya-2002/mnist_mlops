from src.exception import CustomException
from src.logs import logging
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestionConfig

import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

@dataclass
class DataTransformationConfig:
    X_train: str = DataIngestionConfig.X_train
    y_train: str = DataIngestionConfig.y_train
    x_test: str = DataIngestionConfig.x_test
    y_test: str = DataIngestionConfig.y_test


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()


    def initiate_data_transformation(self):
        try:
            logging.info("Initiating Data transformation")
            logging.info("Loading the data from the artifacts folder")
            X_train_path = self.transformation_config.X_train
            y_train_path = self.transformation_config.y_train
            x_test_path = self.transformation_config.x_test
            y_test_path = self.transformation_config.y_test

            X_train = np.load(X_train_path)
            y_train = np.load(y_train_path)
            x_test = np.load(x_test_path)
            y_test = np.load(y_test_path)

            logging.info("Normalizing and Reshaping the data")
            X_train=X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
            X_train=X_train / 255.0
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
            x_test=x_test/255.0
            logging.info("Normalizing and Reshaping the data completed")

            logging.info("Label Encoding the output variables")
            y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
            y_test = tf.one_hot(y_test.astype(np.int32), depth=10)
            logging.info("Label encoding completed")

            return(X_train,y_train,x_test,y_test)
        



        except Exception as e:
            raise CustomException(e,sys)