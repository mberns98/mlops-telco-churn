from .evaluation_utils import evaluate_classification, plot_confusion_matrix

def predict_and_evaluate(model, X_test, y_test, labels=None, verbose=True):
    """
    Performs prediction and evaluates the results.

    Parameters:
        model: Trained classifier.
        X_test (pd.DataFrame): Features to predict.
        y_test (pd.Series): Ground truth labels.
        labels (list, optional): Class labels for plotting.
        verbose (bool): If True, prints metrics.

    Returns:
        dict: Dictionary of classification metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_classification(y_test, y_pred, y_proba=y_proba)

    if verbose:
        print("\n📊 Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

    plot_confusion_matrix(y_test, y_pred, labels=labels)

    return metrics

# Posible uso:
#from models.inference import predict_and_evaluate
#from models.evaluation_utils import plot_roc_curve
#
#metrics = predict_and_evaluate(final_model, X_test, y_test, labels=['Did Not Leave', 'Left'])
#plot_roc_curve(final_model, X_test, y_test)