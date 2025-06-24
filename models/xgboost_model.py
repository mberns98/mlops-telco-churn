import xgboost as xgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Trains an XGBoost model with early stopping and returns the fitted model.
    
    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features for early stopping.
        y_val (pd.Series): Validation labels.

    Returns:
        xgb.XGBClassifier: Trained model.
    """
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        early_stopping_rounds=10,
        eval_metric='aucpr',
        missing=np.nan,
        seed=42
    )

    model.fit(
        X_train, y_train,
        verbose=True,
        eval_set=[(X_val, y_val)]
    )

    return model

def tune_xgboost(X_train, y_train, n_iter=30, cv=3):
    """
    Performs hyperparameter tuning using RandomizedSearchCV and returns the best parameters.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        n_iter (int): Number of iterations.
        cv (int): Cross-validation folds.

    Returns:
        dict: Best hyperparameters found.
    """
    param_dist = {
        'max_depth': randint(3, 6),
        'learning_rate': uniform(0.01, 0.2),
        'gamma': uniform(0, 1),
        'reg_lambda': uniform(0, 10),
        'scale_pos_weight': [1, 3, 5]
    }

    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        seed=42,
        subsample=0.9,
        colsample_bytree=0.5
    )

    opt_random = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        verbose=1,
        n_jobs=-1
    )

    opt_random.fit(X_train, y_train)

    print(f"Best ROC AUC: {opt_random.best_score_:.4f}")
    return opt_random.best_params_

def train_final_model(X_train, y_train, best_params):
    """
    Trains the final XGBoost model using the best parameters.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        best_params (dict): Best parameters from hyperparameter tuning.

    Returns:
        xgb.XGBClassifier: Final trained model.
    """
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        subsample=0.9,
        colsample_bytree=0.5,
        seed=42,
        **best_params
    )

    model.fit(X_train, y_train)
    return model
