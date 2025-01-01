import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import *

def extract_vitals(test_file_name, output_file_name):
    # Load the CSV files
    test_df = pd.read_csv(test_file_name)
    subject_list = test_df['subject_id'].unique()

    # Prepare the list to hold the results
    results = []

    # Iterate through each unique subject_id
    for subject_id in subject_list:
        for t in range(T + 1):
            time_step = t

            # Find the relevant entries in the test DataFrame for the same subject_id
            subject_entries = test_df[test_df['subject_id'] == subject_id].reset_index(drop=True)

            # Identify the chart time for the time step (which is t + 1)
            target_time_step = time_step + 1
            if target_time_step < len(subject_entries):
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

# def main_2(prediction_file, predicted_cols, actual_cols, random_subject_id, predictions_df, results_df):
#     print(f'{extract_part_from_file_path(prediction_file)}')
#     metrics = evaluate_time_series_predictions(predictions_df, results_df, predicted_cols[0], actual_cols[0])  # Just an example
#     print(f'{metrics}')
#     plot_time_series(predictions_df, results_df, random_subject_id, extract_part_from_file_path(prediction_file), predicted_cols, actual_cols)

# def plot_time_series(predictions_dfs, results_df, random_subject_id, model_name, predicted_cols, actual_cols,
#                      subject_id_col='subject_id', time_col='time_step'):
#     # Filter the actual data for the random subject_id
#     subject_actual_data = results_df[results_df[subject_id_col] == random_subject_id]
#
#     plt.figure(figsize=(12, 6))
#
#     # Plot actual values
#     for actual_col in actual_cols:
#         plt.plot(subject_actual_data[time_col], subject_actual_data[actual_col], label=f'Actual {actual_col}', linestyle='-')
#
#     # Plot predicted values for each model
#     for predictions_df, predicted_col in zip(predictions_dfs, predicted_cols):
#         subject_predicted_data = predictions_df[predictions_df[subject_id_col] == random_subject_id]
#         plt.plot(subject_predicted_data[time_col], subject_predicted_data[predicted_col], label=f'Predicted {predicted_col}', linestyle='--')
#
#     plt.title(f'Time Series Plot for Subject ID: {random_subject_id} and Models: {model_name}')
#     plt.xlabel('Time')
#     plt.ylabel('Values')
#     plt.legend()
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

def extract_part_from_file_path(prediction_file):
    # Split the file path by '/'
    parts = prediction_file.split('/')
    if len(parts) > 1:
        second_part = parts[1]
        first_subpart = second_part.split('_')[0]
        if first_subpart == 'random':
            first_subpart = 'random forest'
        return first_subpart
    else:
        return None  # Handle the case where there are not enough parts


def main_2(prediction_file, predicted_cols, actual_cols, random_subject_id, predictions_df, results_df):
    print(f'{extract_part_from_file_path(prediction_file)}')
    metrics = evaluate_time_series_predictions(predictions_df, results_df, predicted_cols[0],
                                               actual_cols[0])  # Example for metrics
    print(f'{metrics}')

    # Plot predictions for the current random subject ID
    plot_time_series(predictions_df, results_df, random_subject_id, extract_part_from_file_path(prediction_file),
                     predicted_cols, actual_cols)


def plot_time_series(predictions_dfs, results_df, random_subject_id, model_name, predicted_cols, actual_cols,
                     subject_id_col='subject_id', time_col='time_step'):
    # Filter the actual data for the random subject_id
    subject_actual_data = results_df[results_df[subject_id_col] == random_subject_id]

    plt.figure(figsize=(12, 6))

    # Plot actual values
    for actual_col in actual_cols:
        plt.plot(subject_actual_data[time_col], subject_actual_data[actual_col], label=f'Actual {actual_col}',
                 linestyle='-')

    # Plot predicted values for each model
    for predictions_df, predicted_col in zip(predictions_dfs, predicted_cols):
        subject_predicted_data = predictions_df[predictions_df[subject_id_col] == random_subject_id]

        if subject_predicted_data.empty:
            print(
                f"No predictions available for subject {random_subject_id} using model {model_name} for {predicted_col}.")
            continue

        # Ensure that the predicted data includes numerical values
        plt.plot(subject_predicted_data[time_col], subject_predicted_data[predicted_col],
                 label=f'Predicted {predicted_col}', linestyle='--')

    plt.title(f'Time Series Plot for Subject ID: {random_subject_id} and Models: {model_name}')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    prediction_files = ['predictions/random_forest_model_predictions.csv',
                        'predictions/ridge_model_predictions.csv',
                        'predictions/xgboost_model_predictions.csv']

    test_file = 'test.csv'  # Input path for test CSV
    output_file = 'actual_vitals.csv'  # Output path for results CSV

    # Load the actual results into a DataFrame
    results_df = pd.read_csv(output_file)

    # Get unique subject IDs and randomly select one for plotting
    unique_subject_ids = results_df['subject_id'].unique()
    random_subject_id = np.random.choice(unique_subject_ids)

    # Load predictions and call the main function for each model
    predictions_dfs = [pd.read_csv(file) for file in prediction_files]
    predicted_cols = ['HeartRate_pred', 'SysBP_pred', 'RespRate_pred', 'SpO2_pred']
    actual_cols = ['HeartRate', 'SysBP', 'RespRate', 'SpO2']

    for i, predictions_df in enumerate(predictions_dfs):
        main_2(prediction_files[i], predicted_cols, actual_cols, random_subject_id, predictions_df, results_df)