import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, file_path='Train.csv', save_split_data=True):
        """
        Initiates the data ingestion process by reading, splitting, and optionally saving the data.

        Parameters:
            file_path (str): Path to the raw data file.
            save_split_data (bool): Flag to save train and test splits to CSV.

        Returns:
            train_set (pd.DataFrame): Training data split.
            test_set (pd.DataFrame): Test data split.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(file_path)
            logging.info(f'Read the dataset as DataFrame with shape: {df.shape}')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Saved raw data to CSV")

            # Train-test split
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")

            # Save train and test sets to CSV if save_split_data is True
            if save_split_data:
                train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
                test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
                logging.info("Saved train and test data splits to CSV")

            logging.info("Data ingestion completed")
            return train_set, test_set

        except Exception as e:
            raise CustomException(e, sys)

        


if __name__ == "__main__":
    # Create an instance of DataIngestion and call initiate_data_ingestion
    data_ingestion = DataIngestion()
    train_df, test_df = data_ingestion.initiate_data_ingestion()  # Returns DataFrames

    # Pass DataFrames directly to DataTransformation
    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test, preprocessor = data_transformation.preprocess_data(train_df, test_df)
    
    # Pass preprocessed data to ModelTrainer for model training
    model_trainer = ModelTrainer()
    model = model_trainer.initiate_model_training(X_train, y_train, X_test, y_test)

    

