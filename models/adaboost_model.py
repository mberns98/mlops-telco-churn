from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import numpy as np

def train_adaboost(X_train, y_train):
    """
    Trains a basic AdaBoost model with decision stumps.

    Returns:
        AdaBoostClassifier: Trained model.
    """
    model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


def tune_adaboost(X_train, y_train, n_iter=20, cv=3):
    """
    Performs hyperparameter tuning for AdaBoost using RandomizedSearchCV.

    Returns:
        dict: Best parameters.
    """
    param_dist = {
        'n_estimators': randint(50, 300),
        'learning_rate': uniform(0.01, 2),
        'base_estimator__max_depth': randint(1, 4)
    }

    base_model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(),
        random_state=42
    )

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
    Trains the final AdaBoost model using the best hyperparameters.

    Returns:
        AdaBoostClassifier: Final trained model.
    """
    model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(),
        **best_params,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model

#Forma de uso:
#from models.adaboost_model import train_adaboost, tune_adaboost, train_final_model
#
#model = train_adaboost(X_train, y_train)
#
#best_params = tune_adaboost(X_train, y_train)
#final_model = train_final_model(X_train, y_train, best_params)