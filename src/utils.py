import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from sklearn.impute import SimpleImputer



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        # Check for NaN or infinity in X_train, y_train, X_test, y_test
        # Handle NaN values using SimpleImputer or other appropriate methods

        # Example: Impute NaN values with the mean
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Convert sparse multilabel-indicator to dense array for y_train
        y_train_dense = y_train.toarray() if hasattr(y_train, 'toarray') else y_train

        # Handle NaN values in y_train
        y_train_dense = imputer.fit_transform(y_train_dense.reshape(-1, 1)).ravel()

        # Convert sparse multilabel-indicator to dense array for y_test
        y_test_dense = y_test.toarray() if hasattr(y_test, 'toarray') else y_test

        # Handle NaN values in y_test
        y_test_dense = imputer.transform(y_test_dense.reshape(-1, 1)).ravel()

        report = {}

        for model_name, model in models.items():
            params = param.get(model_name, {})
            gs = GridSearchCV(model, params, cv=3)

             # Convert sparse matrix X_train to dense array
            X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train

            gs.fit(X_train_dense, y_train_dense)

            # Convert sparse matrix X_test to dense array
            X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train_dense)

            y_train_pred = model.predict(X_train_dense)
            y_test_pred = model.predict(X_test_dense)

            train_model_score = r2_score(y_train_dense, y_train_pred)
            test_model_score = r2_score(y_test_dense, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)