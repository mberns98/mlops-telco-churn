from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix
)
import matplotlib.pyplot as plt

def evaluate_classification(y_true, y_pred, y_proba=None):
    """
    Computes and returns classification metrics.

    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        y_proba (array-like, optional): Probabilities for positive class (for ROC AUC).

    Returns:
        dict: Dictionary with accuracy, precision, recall, f1, and roc_auc (if y_proba provided).
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plots the confusion matrix.

    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        labels (list, optional): Class labels to display.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


def plot_roc_curve(model, X_test, y_test):
    """
    Plots ROC curve for a given model and test data.

    Parameters:
        model: Trained classifier with .predict_proba().
        X_test (pd.DataFrame): Features.
        y_test (pd.Series): True labels.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve")
    plt.grid(True)
    plt.show()