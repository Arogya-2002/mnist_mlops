from src.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig
from src.components.model_evaluation import ModelEvaluation,ModelEvaluationConfig
import pandas as pd


obj=DataIngestion()
data_path=obj.initiate_data_ingestion()
data_transformation_obj = DataTransformation()
X_train,y_train,x_test,y_test = data_transformation_obj.initiate_data_transformation()

model_trainer_obj = ModelTrainer()
x_test,y_test= model_trainer_obj.model_trainer(X_train,y_train,x_test,y_test)

model_evaluation_obj = ModelEvaluation()
model_evaluation_obj.model_evaluation(x_test,y_test)

