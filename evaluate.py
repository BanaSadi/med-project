import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import *
import ast


def extract_vitals(test_file_name, output_file_name):
    # Load the CSV files
    test_df = pd.read_csv(test_file_name)
    subject_list = test_df['subject_id'].unique()

    # Prepare the list to hold the results
    results = []

    # Iterate through each prediction
    for subject_id in subject_list:
        for t in range(T + 1):
            time_step = t

            # Find the relevant entries in the test DataFrame for the same subject_id
            subject_entries = test_df[test_df['subject_id'] == subject_id].reset_index(drop=True)

            # Identify the chart time for the time step (which is t + 1)
            target_time_step = time_step + 1

            # if target_time_step < len(subject_entries):
            vitals_row = subject_entries.iloc[target_time_step]

            result = {
                'subject_id': subject_id,
                'time_step': target_time_step,
                'HeartRate': vitals_row['HeartRate'],
                'SysBP': vitals_row['SysBP'],
                'RespRate': vitals_row['RespRate'],
                'SpO2': vitals_row['SpO2'],
                'age': vitals_row['age'],
                'GENDER_F': vitals_row['GENDER_F'],
                'GENDER_M': vitals_row['GENDER_M']
            }
            results.append(result)

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    results_df.to_csv(output_file_name, index=False)


def evaluate_time_series_predictions(predictions_df, results_df, predicted_col='predicted', actual_col='actual'):
    # Extract true and predicted values
    # y_true = results_df[actual_col]
    # y_pred = predictions_df[predicted_col]
    #
    # # Calculate metrics
    # mae = np.mean(np.abs(y_true - y_pred))
    # mse = np.mean((y_true - y_pred) ** 2)
    # rmse = np.sqrt(mse)
    #
    # # Create a result dictionary
    # metrics = {
    #     'Mean Absolute Error (MAE)': mae,
    #     'Mean Squared Error (MSE)': mse,
    #     'Root Mean Squared Error (RMSE)': rmse
    # }

    # Extract true and predicted values; assuming they are stored as JSON-like string representations
    y_true = np.array(results_df[actual_col])
    y_pred = np.array(predictions_df[predicted_col])

    # Calculate metrics for each dimension
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    rmse = np.sqrt(mse)

    # Create a result dictionary
    metrics = {
        'Mean Absolute Error (MAE)': mae,
        'Mean Squared Error (MSE)': mse,
        'Root Mean Squared Error (RMSE)': rmse
    }

    return metrics


def main_2(prediction_file, predicted_cols, actual_cols, random_subject_id, predictions_df, results_df):
    print(f'{extract_part_from_file_path(prediction_file)}')
    metrics = evaluate_time_series_predictions(predictions_df, results_df, predicted_cols, actual_cols)
    print(f'{metrics}')
    # for predicted_col, actual_col in zip(predicted_cols, actual_cols):
    #     plot_time_series(predictions_df, results_df, random_subject_id, extract_part_from_file_path(prediction_file),
    #                      predicted_col, actual_col)


def plot_time_series(predictions_df, results_df, random_subject_id, model_name, predicted_col='predicted',
                     actual_col='actual', subject_id_col='subject_id', time_col='time_step'):
    # Filter the data for the random subject_id
    subject_actual_data = results_df[results_df[subject_id_col] == random_subject_id]
    subject_predicted_data = predictions_df[predictions_df[subject_id_col] == random_subject_id]

    # Plot the actual vs. predicted time series
    plt.figure(figsize=(12, 6))
    plt.plot(subject_actual_data[time_col], subject_actual_data[f'{actual_col}'], label='Predicted', color='blue',
             linestyle='--')
    plt.plot(subject_predicted_data[time_col], subject_predicted_data[f'{predicted_col}'], label='Actual',
             color='orange')
    plt.title(f'Time Series {actual_col} Plot for Subject ID: {random_subject_id} and model {model_name}')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()


def extract_part_from_file_path(prediction_file):
    # Split the file path by '/'
    parts = prediction_file.split('/')

    # Check if there are at least two parts
    if len(parts) > 1:
        # Take the second part of the split
        second_part = parts[1]

        # Split the second part by '_' and take the first part
        first_subpart = second_part.split('_')[0]
        if first_subpart == 'random':
            first_subpart = 'random forest'
        return first_subpart
    else:
        return None  # Handle the case where there are not enough parts


def plot_all(prediction_files, random_subject_id, results_df, actual_col, predicted_col,
             time_col='time_step', subject_id_col='subject_id'):

    subject_actual_data = results_df[results_df[subject_id_col] == random_subject_id]
    random_forest = pd.read_csv(prediction_files[0])
    ridge = pd.read_csv(prediction_files[1])
    xgboost = pd.read_csv(prediction_files[2])

    subject_predicted_data_forest = random_forest[random_forest[subject_id_col] == random_subject_id]
    subject_predicted_data_ridge = ridge[ridge[subject_id_col] == random_subject_id]
    subject_predicted_data_xg = xgboost[xgboost[subject_id_col] == random_subject_id]

    # Plot the actual vs. predicted time series
    plt.figure(figsize=(12, 6))
    plt.plot(subject_actual_data[time_col], subject_actual_data[f'{actual_col}'], label='Actual', color='blue',
             linestyle='--')
    plt.plot(subject_predicted_data_forest[time_col], subject_predicted_data_forest[f'{predicted_col}'],
             label=f'{extract_part_from_file_path(prediction_files[0])}',
             color='orange')
    plt.plot(subject_predicted_data_ridge[time_col], subject_predicted_data_ridge[f'{predicted_col}'],
             label=f'{extract_part_from_file_path(prediction_files[1])}',
             color='red')
    plt.plot(subject_predicted_data_xg[time_col], subject_predicted_data_xg[f'{predicted_col}'],
             label=f'{extract_part_from_file_path(prediction_files[2])}',
             color='green')
    plt.title(f'Time Series {actual_col} Plot for Subject ID: {random_subject_id}', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Values', fontsize=14)
    plt.legend()
    plt.xticks()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    prediction_files = ['predictions/random_forest_model_predictions.csv', 'predictions/ridge_model_predictions.csv',
                        'predictions/xgboost_model_predictions.csv']

    test_file = 'test.csv'  # Input path for test CSV
    output_file = 'actual_vitals.csv'  # Output path for results CSV
    # extract_vitals(test_file, output_file)
    predicted_cols = ['HeartRate_pred', 'SysBP_pred', 'RespRate_pred', 'SpO2_pred']
    actual_cols = ['HeartRate', 'SysBP', 'RespRate', 'SpO2']
    output_file_name = 'actual_vitals.csv'
    # Load the data
    results_df = pd.read_csv(output_file_name)
    unique_subject_ids = results_df['subject_id'].unique()
    random_subject_id = np.random.choice(unique_subject_ids)
    num = 1
    random_subject_ids = np.random.choice(unique_subject_ids, num, replace=False)
    # for file in prediction_files:
    #     predictions_df = pd.read_csv(file)
    #     main_2(file, predicted_cols, actual_cols, random_subject_id, predictions_df, results_df)
    for col, pred_col in zip(actual_cols, predicted_cols):
        for random_subject_id in random_subject_ids:
            plot_all(prediction_files, random_subject_id, results_df, col, pred_col)
