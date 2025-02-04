from src.exception import CustomException
from src.logs import logging
from dataclasses import dataclass
from src.constants import INPUT_SHAPE, BATCH_SIZE, NUM_CLASSES, EPOCHS
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.keras')
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.RMSprop(epsilon=1e-08)
    loss: str = 'categorical_crossentropy'


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') > 0.995:
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.callbacks = myCallback()

    def build_model(self):
        try:
            logging.info("Building the CNN model")
            model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=INPUT_SHAPE),  # Use Input layer here
                tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.MaxPool2D(strides=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')])
            logging.info("Model built successfully")
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def model_trainer(self, X_train, y_train, x_test, y_test):
        logging.info("Compiling the model")

            # Log the model summary to MLflow
        model = self.build_model()
        model_summary_str = []
        model.summary(print_fn=lambda x: model_summary_str.append(x))
        mlflow.log_param("model_architecture", "\n".join(model_summary_str))

        model.compile(optimizer=self.model_trainer_config.optimizer, loss=self.model_trainer_config.loss, metrics=['acc'])

        history = model.fit(X_train, y_train,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                validation_split=0.1,
                                callbacks=[self.callbacks])

            # Log metrics for each epoch

            # Log final metrics
        final_loss = history.history['loss'][-1]
        final_acc = history.history['acc'][-1]

        logging.info(f"Model saved successfully at: {self.model_trainer_config.trained_model_file_path}")
        model.save(self.model_trainer_config.trained_model_file_path)

        return x_test, y_test
