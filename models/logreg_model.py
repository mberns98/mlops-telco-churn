import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import numpy as np

def train_logistic_regression(X_train, y_train):
    """
    Trains a basic Logistic Regression model.

    Returns:
        LogisticRegression: Trained model.
    """
    model = LogisticRegression(
        solver='liblinear',  # Bueno para datasets pequeños/medianos y L1/L2
        penalty='l2',
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


def tune_logistic_regression(X_train, y_train, n_iter=20, cv=3):
    """
    Performs hyperparameter tuning for Logistic Regression using RandomizedSearchCV.

    Returns:
        dict: Best parameters.
    """
    param_dist = {
        'C': uniform(0.01, 10),         # Regularización inversa
        'penalty': ['l1', 'l2'],        # Tipos de regularización
        'solver': ['liblinear']        # Solo solver compatible con l1 y l2
    }

    base_model = LogisticRegression(random_state=42, max_iter=1000)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    print(f"Best ROC AUC: {search.best_score_:.4f}")
    return search.best_params_


def train_final_model(X_train, y_train, best_params):
    """
    Trains the final Logistic Regression model with the best hyperparameters.

    Returns:
        LogisticRegression: Final trained model.
    """
    model = LogisticRegression(
        **best_params,
        random_state=42,
        max_iter=1000
    )

    model.fit(X_train, y_train)
    return model

# Forma de uso:
#from models.logreg_model import train_logistic_regression, tune_logistic_regression, train_final_model
#
## Entrenamiento rápido (sin tuning)
#model = train_logistic_regression(X_train, y_train)
#
## Tuning
#best_params = tune_logistic_regression(X_train, y_train)
#final_model = train_final_model(X_train, y_train, best_params)