from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, f1_score)
import numpy as np

def get_isnull_ratio(df):
    return (df.isnull().sum() / len(df)).sort_values(ascending=False)

def eval_metrics(y_true, y_pred, y_prob=None, type="classification"):
    if type == "classification":
        try:
            roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else roc_auc_score(y_true, y_pred)
            roc_auc_fpr001 = roc_auc_score(y_true, y_prob, max_fpr=0.01) if y_prob is not None else roc_auc_score(y_true, y_pred, max_fpr=0.01)
        except Exception:
            roc_auc = float("nan")
            roc_auc_fpr001 = float("nan")
        return {
            "ACC": accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred, average="binary"),
            "BAL-ACC": balanced_accuracy_score(y_true, y_pred),
            "ROC-AUC": roc_auc,
            "ROC-AUC@0.01FPR": roc_auc_fpr001,
        }
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)).item(),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
