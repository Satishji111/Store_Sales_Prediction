import os
import sys
import numpy as np
import math
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class ModelTrainerConfig:
    model_save_path = os.path.join('artifacts', 'best_model.pkl')

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        
        # Define models with polynomial feature pipeline where needed
        self.models = {
            'LinearRegression': Pipeline([
                ('scaler', StandardScaler(with_mean=False)),
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ]),
            'Lasso': Pipeline([
                ('scaler', StandardScaler(with_mean=False)),
                ('poly', PolynomialFeatures(degree=2)),
                ('lasso', Lasso())
            ]),
            'Ridge': Pipeline([
                ('scaler', StandardScaler(with_mean=False)),
                ('poly', PolynomialFeatures(degree=2)),
                ('ridge', Ridge())
            ]),
            'RandomForest': RandomForestRegressor(),
            'XGBoost': XGBRegressor()
        }

        # Parameter grid for models that support hyperparameter tuning
        self.param_grid = {
            'Lasso': {'lasso__alpha': [0.1, 0.5, 1.0, 2.0]},
            'Ridge': {'ridge__alpha': [0.1, 1.0, 10.0]},
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7]
            }
        }

    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """
        Train and evaluate a model, logging metrics.
        """
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        rmse=math.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {'mse': mse,'rmse':rmse, 'mae': mae, 'r2_score': r2}

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        """
        Trains and selects the best model based on R2 score.
        """
        try:
            best_model = None
            best_score = np.inf
            best_metrics = {}

            for model_name, model in self.models.items():
                logging.info(f"Training {model_name} model.")
                
                if model_name in self.param_grid:
                    # Hyperparameter tuning
                    grid_search = GridSearchCV(estimator=model, param_grid=self.param_grid[model_name],
                                               cv=5, n_jobs=-1, scoring='r2')
                    grid_search.fit(X_train, y_train)
                    model = grid_search.best_estimator_
                    logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
                
                # Evaluate the model
                metrics = self.evaluate_model(model, X_train, X_test, y_train, y_test)
                logging.info(f"{model_name} metrics: {metrics}")

                if metrics['rmse'] < best_score:
                    best_score = metrics['rmse']
                    best_model = model
                    best_metrics = metrics

            # Save the best model
            save_object(self.config.model_save_path, best_model)
            logging.info(f"Best model {best_model} saved with RMSE value : {best_score}")
            
            return best_model, best_metrics

        except Exception as e:
            logging.error(f"Model training error: {e}")
            raise CustomException(e, sys)

