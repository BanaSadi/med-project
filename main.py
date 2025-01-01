import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import xgboost as xgb  # Import XGBoost
from config import T
import matplotlib.pyplot as plt
import random

def encode_demographic_data(data, demographic_cols):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_demographics = encoder.fit_transform(data[demographic_cols])
    encoded_cols = encoder.get_feature_names_out(demographic_cols)
    encoded_df = pd.DataFrame(encoded_demographics, columns=encoded_cols, index=data.index)
    return pd.concat([data.drop(columns=demographic_cols), encoded_df], axis=1), encoder


def split_data(df):
    subject_ids = df['subject_id'].unique()
    train_subjects, test_subjects = train_test_split(subject_ids, test_size=0.2, random_state=42)
    train_data = df[df['subject_id'].isin(train_subjects)]
    test_data = df[df['subject_id'].isin(test_subjects)]
    return train_data, test_data


def prepare_features_with_demographics(data, target_cols, demographic_cols):
    features, targets, subject_ids = [], [], []
    grouped = data.groupby('subject_id')

    for subject_id, group in grouped:
        group = group.sort_values('charttime')  # Sort the group's entries by time
        demographics = group.iloc[0][demographic_cols].values

        for i in range(1, T):
            past_data = group.iloc[:i][target_cols].values.flatten()
            feature = np.hstack([demographics, past_data])

            features.append(feature)
            targets.append(group.iloc[i][target_cols].values)
            subject_ids.append(subject_id)  # Append the current subject_id

    return features, np.array(targets), subject_ids


def pad_features(features, max_len):
    return np.array([np.pad(f, (0, max_len - len(f)), constant_values=0) for f in features])


def save_model(model, model_name):
    joblib.dump(model, f"{model_name}.joblib")
    print(f"Model saved as {model_name}.joblib")


def load_model(model_name):
    if os.path.exists(f"{model_name}.joblib"):
        print(f"Loading model {model_name}.joblib")
        return joblib.load(f"{model_name}.joblib")
    else:
        return None


def predict_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, demographic_cols, subjects):
    # Check if previously saved model exists
    existing_model = load_model(model_name)
    if existing_model:
        model = existing_model
    else:
        model.fit(X_train, y_train)
        save_model(model, model_name)

    # Make predictions
    predictions = model.predict(X_test)


    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

    # Print metrics
    print(f"Model: {model.__class__.__name__}, MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")

    # Create a DataFrame to store predictions with subject_id, demographic info and time steps
    prediction_rows = []
    # print(X_test, subjects)
    # print(predictions)

    # Assume X_test also contains column for subject_id to retrieve demographic info.
    for idx, subject_id in enumerate(subjects):
        t = (idx % (T - 1)) + 1
        row = {
            'subject_id': subject_id,
            'time_step': t,
            'HeartRate_pred': predictions[idx][0],
            'SysBP_pred': predictions[idx][1],
            'RespRate_pred': predictions[idx][2],
            'SpO2_pred': predictions[idx][3],
        }
        prediction_rows.append(row)

    # Create a predictions DataFrame
    predictions_df = pd.DataFrame(prediction_rows)

    # Save predictions to a CSV file
    predictions_df.to_csv(f"{model_name}_predictions.csv", index=False)
    print(f"Predictions saved to {model_name}_predictions.csv")


    # Residual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, predictions - y_test, color='blue', s=10, label='Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residuals vs Predicted for {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()

    # Prediction vs Actual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, color='blue', label='Predictions')
    plt.plot(y_test, y_test, color='red', label='Perfect Prediction')
    plt.title(f'Predicted vs Actual for {model_name}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()

    return predictions_df


def plot_subject_ts_and_predictions(subject_id, test_data, predictions_df, target_cols):
    # Filter the test data for the given subject
    subject_test_data = test_data[test_data['subject_id'] == subject_id].sort_values('charttime')

    # Filter the predictions DataFrame for the given subject
    subject_predictions = predictions_df[predictions_df['subject_id'] == subject_id]

    # Ensure we have enough time steps
    if len(subject_test_data) < len(target_cols) + 1 or len(subject_predictions) < len(target_cols):
        print(f"Not enough time steps for subject {subject_id} (found test: {len(subject_test_data)}, "
              f"found predictions: {len(subject_predictions)})")
        return

    # Prepare actual values from the test data
    actual_values = subject_test_data[target_cols].values.flatten()

    # Prepare predicted values from predictions DataFrame
    predicted_values = subject_predictions[[f'{col}_pred' for col in target_cols]].values.flatten()

    # Make sure lengths are matching for plotting
    time_steps = range(len(actual_values))  # Time steps based on actual values

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot actual vs predicted values
    for i, target_col in enumerate(target_cols):
        plt.subplot(len(target_cols), 1, i + 1)
        plt.plot(time_steps, actual_values, label='Actual', marker='o', color='blue', linestyle='solid')
        plt.plot(time_steps[:-1], predicted_values[:-1], label='Predicted', marker='x', color='red', linestyle='dashed')
        plt.title(f'Time Series and Predictions for Subject ID: {subject_id} - {target_col}')
        plt.xlabel('Time Steps')
        plt.ylabel(target_col)
        plt.axvline(x=len(actual_values) - 1, color='green', linestyle='--', label='Prediction Start')
        plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    final_data = pd.read_csv('final_data_27_12.csv')
    final_data = final_data[['subject_id', 'charttime', 'HeartRate', 'SysBP', 'RespRate', 'SpO2', 'GENDER', 'age']]
    demographic_cols_encode = ['GENDER']
    target_cols = ['HeartRate', 'SysBP', 'RespRate', 'SpO2']
    # Group by 'subject_id' and calculate the number of charttime points per subject
    charttime_length_per_subject = final_data.groupby('subject_id')['charttime'].count()
    subjects_with_72_or_more = charttime_length_per_subject[charttime_length_per_subject >= 72].index
    # Filter the data to include only these subjects
    final_data = final_data[final_data['subject_id'].isin(subjects_with_72_or_more)]

    # Print the result
    print(f"Filtered dataset shape: {final_data.shape}")
    print(f"Number of unique subjects remaining: {final_data['subject_id'].nunique()}")
    print("finished loading data")

    # Encode data
    encoded_data, encoder = encode_demographic_data(final_data, demographic_cols_encode)

    # Split data
    train_data, test_data = split_data(encoded_data)

    # Prepare features with demographics
    demographic_cols = list(set(encoded_data.columns) - set(target_cols + ['subject_id', 'charttime']))

    print(demographic_cols)

    X_train, y_train, train_subject_ids = prepare_features_with_demographics(train_data, target_cols, demographic_cols)
    X_test, y_test, test_subject_ids = prepare_features_with_demographics(test_data, target_cols, demographic_cols)

    print("data encoded and split")
    # print("X looks like")
    # print(X_test)
    # print("y looks like")
    # print(y_test)
    # print("subjects look like")
    # print(test_subject_ids)

    # Pad features
    max_len = max(max(len(f) for f in X_train), max(len(f) for f in X_test))
    print(f'max len id {max_len}')
    X_train = pad_features(X_train, max_len)
    X_test = pad_features(X_test, max_len)
    # print("X looks like")
    # print(X_test)

    # Save test DataFrame to CSV
    test_data.to_csv('test.csv', index=False)

    print("padding ended")

    # Models with their respective filenames
    models = [
        (RandomForestRegressor(n_estimators=100, random_state=42), "random_forest_model"),
        (Ridge(alpha=1.0), "ridge_model"),
        (xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42), "xgboost_model")
    ]

    for model, model_name in models:
        print(f" training {model_name}")
        predictions_df = predict_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, demographic_cols, test_subject_ids)

        # # Select 3 random subject IDs from the training data
        # unique_subject_ids = test_data['subject_id'].unique()
        # random_subject_ids = random.sample(list(unique_subject_ids), 1)
        # print('printing examples')
        # # After evaluating models and saving predictions
        # for subject_id in random_subject_ids:
        #     plot_subject_ts_and_predictions(subject_id, test_data, predictions_df, target_cols)



if __name__ == "__main__":
    main()