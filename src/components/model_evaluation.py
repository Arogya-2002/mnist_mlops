from src.exception import CustomException
from src.logs import logging
from dataclasses import dataclass
import os
import sys
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import mlflow
import mlflow.keras  # For logging Keras models

@dataclass
class ModelEvaluationConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.keras')

class ModelEvaluation:
    def __init__(self):
        self.model_evaluation_config = ModelEvaluationConfig()

    def model_evaluation(self, x_test, y_test):
        try:
                # Load the trained model
                loaded_model = tf.keras.models.load_model(self.model_evaluation_config.trained_model_file_path)
                
                # Evaluate the model on the test data
                test_loss, test_acc = loaded_model.evaluate(x_test, y_test)
                print(f"The test accuracy is: {test_acc}")
                print(f"The test loss is: {test_loss}")
                
                # Log the metrics to MLflow
                # mlflow.log_metric("test_loss", test_loss)
                # mlflow.log_metric("test_accuracy", test_acc)

                # Generate predictions on the test data
                y_pred = loaded_model.predict(x_test)
                
                # Convert predictions from probabilities to class labels
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true = np.argmax(y_test, axis=1)

                # Compute the confusion matrix
                conf_matrix = confusion_matrix(y_true, y_pred_classes)
                
                # Print the confusion matrix
                print("Confusion Matrix:")
                print(conf_matrix)

                # Log the confusion matrix as an artifact
                conf_matrix_path = "confusion_matrix.txt"
                with open(conf_matrix_path, "w") as f:
                    f.write(str(conf_matrix))
                
                # mlflow.log_artifact(conf_matrix_path)

                # Log the trained model to MLflow (optional for reusability)
                # mlflow.keras.log_model(loaded_model, "model")

        except Exception as e:
            raise CustomException(e, sys)
