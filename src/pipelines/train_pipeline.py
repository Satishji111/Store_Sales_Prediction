import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

def run_training_pipeline():
    try:
        logging.info("Starting the training pipeline...")

        # Step 1: Data Ingestion
        logging.info("Running data ingestion...")
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        logging.info("Running data transformation...")
        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test, preprocessor = data_transformation.preprocess_data(train_data, test_data)

        # Step 3: Model Training
        logging.info("Running model training...")
        model_trainer = ModelTrainer()
        best_model, best_metrics = model_trainer.initiate_model_training(X_train, y_train, X_test, y_test)

        logging.info("Training pipeline completed successfully.")
        logging.info(f"Best Model: {best_model}")
        logging.info(f"Best Model Metrics: {best_metrics}")

    except Exception as e:
        logging.error("Error during the training pipeline.")
        raise CustomException(e, sys)

