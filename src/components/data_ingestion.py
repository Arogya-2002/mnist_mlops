from src.exception import CustomException
from src.logs import logging

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    dir_path: str = os.path.join('artifacts','dataset')
    X_train: str = os.path.join('artifacts','dataset','mnist_x_train.npy')
    y_train: str = os.path.join('artifacts','dataset','mnist_y_train.npy')
    x_test: str = os.path.join('artifacts','dataset','mnist_x_test.npy')
    y_test: str = os.path.join('artifacts','dataset','mnist_y_test.npy')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiating data ingestion")
        try:
           #load the MNIST data set
            logging.info("Saving the files into the artifacts folder")
            mnist = tf.keras.datasets.mnist
            (X_train, y_train), (x_test, y_test) = mnist.load_data() 
            os.makedirs(self.ingestion_config.dir_path, exist_ok=True)
            np.save(self.ingestion_config.X_train, X_train)
            np.save(self.ingestion_config.y_train, y_train)
            np.save(self.ingestion_config.x_test, x_test)
            np.save(self.ingestion_config.y_test, y_test)
            logging.info("Successfully saved the MNIST data into the artifacts folder")
        except Exception as e:
            raise CustomException(e,sys)
