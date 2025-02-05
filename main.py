from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.model_evaluation import ModelEvaluation, ModelEvaluationConfig
from src.logs import logging
from src.exception import CustomException
import sys

def main():
    try:
        # Data Ingestion
        logging.info("Starting data ingestion process...")
        obj = DataIngestion()
        data_path = obj.initiate_data_ingestion()
        logging.info(f"Data ingestion completed. Data available at: {data_path}")
        
        # Data Transformation
        logging.info("Starting data transformation process...")
        data_transformation_obj = DataTransformation()
        X_train, y_train, x_test, y_test = data_transformation_obj.initiate_data_transformation()
        logging.info("Data transformation completed.")
        
        # Model Training
        logging.info("Starting model training process...")
        model_trainer_obj = ModelTrainer()
        x_test, y_test = model_trainer_obj.model_trainer(X_train, y_train, x_test, y_test)
        logging.info("Model training completed and model saved.")
        
        # Model Evaluation
        logging.info("Starting model evaluation process...")
        model_evaluation_obj = ModelEvaluation()
        model_evaluation_obj.model_evaluation(x_test, y_test)
        logging.info("Model evaluation completed.")
        
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == "__main__":
    main()
