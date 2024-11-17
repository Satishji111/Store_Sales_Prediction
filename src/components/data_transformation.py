import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    transformed_train_data_path = os.path.join('artifacts', 'transformed_train.pkl')
    transformed_test_data_path = os.path.join('artifacts', 'transformed_test.pkl')
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

        self.categorical_features = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type' , 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

    def map_inaccurate_values(self, df):
        # Mapping for other categorical features
        mappings = {
            'Item_Fat_Content': {
                'Low Fat': 'Low Fat',
                'low fat': 'Low Fat',
                'LF': 'Low Fat',
                'Regular': 'Regular',
                'reg': 'Regular'
            }
        }
        
        # Apply the mappings for each column
        for column, map_dict in mappings.items():
            if column in df.columns:
                df[column] = df[column].replace(map_dict)
    
        # Transform Item_Identifier by taking the first two characters
        if 'Item_Identifier' in df.columns:
            df['Item_Identifier'] = df['Item_Identifier'].str[:2]
      
        logging.info("Inaccurate values and ordinal mappings completed.")
        return df

        
    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        try:
            # Map inaccurate values
            train_df = self.map_inaccurate_values(train_df)
            test_df = self.map_inaccurate_values(test_df)

            # Dynamically handle year if not provided
            current_year = pd.Timestamp.now().year
            if 'Outlet_Establishment_Year' in train_df.columns:
                train_df['Outlet_Age'] = current_year - train_df['Outlet_Establishment_Year']
                test_df['Outlet_Age'] = current_year - test_df['Outlet_Establishment_Year']
                logging.info("Transformed 'Outlet_Establishment_Year' to 'Outlet_Age'.")

            # Separate features and target
            target_column = 'Item_Outlet_Sales'
            y_train = train_df[target_column]
            y_test = test_df[target_column]
            X_train = train_df.drop(columns=[target_column, 'Outlet_Identifier','Outlet_Establishment_Year'], axis=1)
            X_test = test_df.drop(columns=[target_column, 'Outlet_Identifier','Outlet_Establishment_Year'], axis=1)

            logging.info(f"X_train shape after dropping: {X_train.shape}")
            logging.info(f"X_train columns after dropping: {X_train.columns}")
            logging.info(f"y_train shape: {y_train.shape}")
            logging.info(f"X_test shape after dropping: {X_test.shape}")
            logging.info(f"y_test shape: {y_test.shape}")

             # Ensure that the lengths match
            if len(X_train) != len(y_train):
                raise ValueError(f"Length mismatch: X_train has {len(X_train)} samples, y_train has {len(y_train)} samples.")
            # Define numerical, categorical, and ordinal feature sets
            numerical_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']
            categorical_features = self.categorical_features


            # Define pipelines for numerical, categorical, and ordinal transformations
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(sparse=False)),
                ("scaler",StandardScaler(with_mean=False))
            ])

            # Column transformer combining all pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_features),
                    ('cat', categorical_pipeline, categorical_features),
                ]
            )

            # Apply transformations
            logging.info("Applying preprocessing to the data.")
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            # Log shapes after transformation
            logging.info(f"Transformed X_train shape: {X_train.shape}")
            logging.info(f"Transformed X_test shape: {X_test.shape}")
            # Convert to dense arrays if necessary (safety step for sparse outputs)
            X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
            X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

            logging.info(f"Train DataFrame columns: {train_df.columns.tolist()}")
            logging.info(f"Test DataFrame columns: {test_df.columns.tolist()}")


            # Save transformed data and preprocessor object
            save_object(self.transformation_config.transformed_train_data_path, X_train)
            save_object(self.transformation_config.transformed_test_data_path, X_test)
            save_object(self.transformation_config.preprocessor_obj_file_path, preprocessor)

            logging.info("Data transformation and saving completed successfully.")
            return X_train, X_test, y_train, y_test, preprocessor

        except Exception as e:
            logging.error(f"Data transformation error: {e}")
            raise CustomException(e, sys)
