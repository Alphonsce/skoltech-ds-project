import numpy as np
import pandas as pd
import optuna
from typing import Optional, Union, List
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

import sys
sys.path.append("..")
sys.path.append(".")

from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from src.utils import eval_metrics
from src.imputers import fill_nulls_with_mean, impute_with_boosting

def best_trial_callback(study, trial):
    if study.best_trial.number == trial.number:
        print(f"🏆 New Best Trial {trial.number}; Score: {trial.value:.5f}")

def objective(trial, X_train, X_val, y_train, y_val, cat_features, score_metric="roc_auc", use_gpu=False):
    params = {
        "iterations": trial.suggest_int("iterations", 700, 1300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "depth": trial.suggest_int("depth", 6, 12),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-1, 10.0, log=True),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 3.0, 6.0),
        "verbose": False,
    }
    
    if use_gpu:
        params["task_type"] = "GPU"
        params["devices"] = "0"
    
    model = CatBoostClassifier(**params)
    
    fit_kwargs = {}
    if cat_features is not None:
        fit_kwargs["cat_features"] = cat_features

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
        **fit_kwargs
    )
    
    y_pred = model.predict(X_val)

    print(f"Using {score_metric} as score metric")

    if score_metric == "roc_auc":
        score = roc_auc_score(y_val, y_pred)
    elif score_metric == "roc_auc_001":
        score = roc_auc_score(y_val, y_pred, max_fpr=0.01)
    elif score_metric == "balanced_accuracy":
        score = balanced_accuracy_score(y_val, y_pred)
    elif score_metric == "f1":
        score = f1_score(y_val, y_pred, average="binary")
    else:
        raise ValueError(f"Invalid score metric: {score_metric}")
    
    return score

def optimize_catboost_optuna(X_train, X_val, y_train, y_val, score_metric="roc_auc", cat_features=None, n_trials=100, use_gpu=False):
    study = optuna.create_study(
        direction="maximize",
        study_name="catboost_optimization",
        pruner=optuna.pruners.HyperbandPruner()
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, X_val, y_train, y_val, cat_features, score_metric, use_gpu),
        n_trials=n_trials,
        callbacks=[best_trial_callback],
        show_progress_bar=True
    )
    
    return study

def train_and_create_best_submission(X, y, X_test, df_submission, best_params, cat_features=None, use_gpu=False, verbose=100, submission_path="submissions/submission_catboost.csv"):
    print(f"Training with best params for submission:...")
    if use_gpu:
        best_params["task_type"] = "GPU"
        best_params["devices"] = "0"

    model = CatBoostClassifier(**best_params)

    fit_kwargs = {"verbose": verbose}
    if cat_features is not None:
        fit_kwargs["cat_features"] = cat_features

    model.fit(X, y, **fit_kwargs)

    y_pred = model.predict(X_test)
    df_submission["CHURN"] = y_pred
    df_submission.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    return model, y_pred, df_submission

def cross_validate_catboost(
    X: pd.DataFrame, y: pd.Series,
    best_params: dict,
    imputation_method,    # "mean" or "catboost"
    numerical_cols_to_fill,
    cat_cols_to_fill,
    cat_features=None,
    use_gpu=False,
    cv=5,
    random_state=42,
    verbose=100,
):
    params = best_params.copy()

    if use_gpu:
        params["task_type"] = "GPU"
        params["devices"] = "0"

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold, (train_idx, val_idx) in tqdm(enumerate(skf.split(X, y), 1), total=cv, desc="Cross-Validating, Fold:..."):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if imputation_method == "mean":
            X_train = fill_nulls_with_mean(
                df_to_fill=X_train,
                df_to_compute_mean=X_train,
                numerical_columns=numerical_cols_to_fill,
                categorical_columns=cat_cols_to_fill
            )

            X_val = fill_nulls_with_mean(
                df_to_fill=X_val,
                df_to_compute_mean=X_train,
                numerical_columns=numerical_cols_to_fill,
                categorical_columns=cat_cols_to_fill
            )

        elif imputation_method == "catboost":
            for col_name in cat_cols_to_fill:
                print(f"Imputing '{col_name}' ")
                X_train, X_val, model = impute_with_boosting(
                    df_to_train_on=X_train,
                    df_train_to_impute=X_train,
                    df_test_to_impute=X_val,
                    column_name=col_name,
                    column_type="categorical",
                    feature_columns=None,
                    use_gpu=True,
                    verbose=False,
                )

            for col_name in numerical_cols_to_fill:
                print(f"Imputing '{col_name}' ")
                X_train, X_val, model = impute_with_boosting(
                    df_to_train_on=X_train,
                    df_train_to_impute=X_train,
                    df_test_to_impute=X_val,
                    column_name=col_name,
                    column_type="numerical",
                    feature_columns=numerical_cols_to_fill,
                    use_gpu=True,
                    verbose=False,
                )

        else:
            raise ValueError(f"Invalid imputation method: {imputation_method}")

        model = CatBoostClassifier(**params)

        fit_kwargs = {"verbose": verbose}
        if cat_features is not None:
            fit_kwargs["cat_features"] = cat_features

        model.fit(X_train, y_train, **fit_kwargs)
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]

        metrics = eval_metrics(y_val, y_val_pred, y_val_pred_proba, type="classification")
        fold_metrics.append(metrics)

    agg_mean = {}
    agg_std = {}

    for metric_name in fold_metrics[0].keys():
        vals = [m.get(metric_name, np.nan) for m in fold_metrics]
        agg_mean[metric_name] = np.mean(vals).item()
        agg_std[metric_name] = np.std(vals).item()

    result = {
        "folds": fold_metrics,
        "metrics_mean": agg_mean,
        "metrics_std": agg_std,
    }

    return result
