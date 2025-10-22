from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

def fill_nulls_with_mean(
    df_to_fill, df_to_compute_mean, numerical_columns, categorical_columns=None
):
    df_to_fill = df_to_fill.copy()
    for col in numerical_columns:
        df_to_fill[col].fillna(df_to_compute_mean[col].mean(), inplace=True)
    
    if categorical_columns is not None:
        for col in categorical_columns:
            mode_value = df_to_compute_mean[col].mode()[0] if not df_to_compute_mean[col].mode().empty else None
            if mode_value is not None:
                df_to_fill[col].fillna(mode_value, inplace=True)
    
    return df_to_fill

def impute_with_boosting(df_to_train_on, df_train_to_impute, df_test_to_impute, column_name, column_type, feature_columns=None, model_params=None, use_gpu=True, verbose=200):
    df_train = df_to_train_on.copy()
    df_train_impute = df_train_to_impute.copy()
    df_test_impute = df_test_to_impute.copy()
    
    if feature_columns is None:
        feature_columns = [col for col in df_train.columns if col != column_name]
    
    categorical_features = [
        col for col in feature_columns 
        if (df_train[col].dtype == 'object' or df_train[col].dtype.name == 'category') and col != column_name
    ]
    
    train_mask = df_train[column_name].notna()
    train_data = df_train[train_mask]
    
    default_params = {
        'random_state': 42,
        'verbose': verbose,
        'thread_count': -1,
    }
    
    if use_gpu:
        default_params.update({
            'task_type': 'GPU',
            'devices': '0',
        })
    
    if model_params:
        default_params.update(model_params)
    
    if column_type == 'categorical':
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(train_data[column_name])
        
        X_train = train_data[feature_columns]
        
        model = CatBoostClassifier(**default_params)
        model.fit(
            X_train, y_train_encoded,
            cat_features=categorical_features,
        )
        
        train_impute_mask = df_train_impute[column_name].isna()
        test_impute_mask = df_test_impute[column_name].isna()
        
        if train_impute_mask.sum() > 0:
            X_train_impute = df_train_impute[train_impute_mask][feature_columns]
            predicted_encoded_train = model.predict(X_train_impute)
            predicted_values_train = le.inverse_transform(predicted_encoded_train)
            df_train_impute.loc[train_impute_mask, column_name] = predicted_values_train
        
        if test_impute_mask.sum() > 0:
            X_test_impute = df_test_impute[test_impute_mask][feature_columns]
            predicted_encoded_test = model.predict(X_test_impute)
            predicted_values_test = le.inverse_transform(predicted_encoded_test)
            df_test_impute.loc[test_impute_mask, column_name] = predicted_values_test
        
        train_accuracy = model.score(X_train, y_train_encoded)
        print(f"Training accuracy: {train_accuracy:.4f}")
        
    elif column_type == 'numerical':
        y_train = train_data[column_name]
        X_train = train_data[feature_columns]
        
        model = CatBoostRegressor(**default_params)
        model.fit(
            X_train, y_train,
            cat_features=categorical_features,
        )
        
        train_impute_mask = df_train_impute[column_name].isna()
        test_impute_mask = df_test_impute[column_name].isna()
        
        if train_impute_mask.sum() > 0:
            X_train_impute = df_train_impute[train_impute_mask][feature_columns]
            predicted_values_train = model.predict(X_train_impute)
            df_train_impute.loc[train_impute_mask, column_name] = predicted_values_train
        
        if test_impute_mask.sum() > 0:
            X_test_impute = df_test_impute[test_impute_mask][feature_columns]
            predicted_values_test = model.predict(X_test_impute)
            df_test_impute.loc[test_impute_mask, column_name] = predicted_values_test
        
        train_predictions = model.predict(X_train)
        mae = np.mean(np.abs(train_predictions - y_train))
        rmse = np.sqrt(np.mean((train_predictions - y_train) ** 2))
        print(f"Training MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    return df_train_impute, df_test_impute, model