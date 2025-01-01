import pandas as pd


def calculate_severity(df, normal_ranges):
    # Calculate severity for each vital
    for column, (lower, upper) in normal_ranges.items():
        # Count of measurements outside the normal range
        df['severity_score_1'] += ((df[column] < lower) | (df[column] > upper)).astype(int)

        # Calculate how extreme the measurement is
        df[column + '_extreme'] = df[column].apply(lambda x: max(x - upper, lower - x, 0))

        # Min-Max normalization for the extreme measure to scale between 0 and 4
        min_value = df[column + '_extreme'].min()
        max_value = df[column + '_extreme'].max()

        # Normalize and scale to 0-4
        df[column + '_severity_normalized'] = (
            (df[column + '_extreme'] - min_value) / (max_value - min_value)
            if (max_value - min_value) != 0 else 0
        )

        # Add normalized severity score to the total severity score
        df['severity_score_2'] += df[column + '_severity_normalized']

    return df  # Corrected indentation


def check_severity_range(df, score_1_column='severity_score_1', score_2_column='severity_score_2'):
    # Group by severity_score_1 and calculate min and max for severity_score_2
    severity_ranges = df.groupby(score_1_column)[score_2_column].agg(['min', 'max']).reset_index()

    # Rename columns for clarity
    severity_ranges.columns = [score_1_column, 'min_' + score_2_column, 'max_' + score_2_column]

    print(severity_ranges)


def main():
    # Define normal ranges for the vitals
    normal_ranges = {
        'HeartRate_pred': (60, 100),  # Example normal range for heart rate
        'SysBP_pred': (90, 120),  # Example normal range for systolic blood pressure
        'RespRate_pred': (12, 20),  # Example normal range for respiratory rate
        'SpO2_pred': (95, 100)  # Example normal range for SpO2
    }

    normal_ranges_2 = {
        'HeartRate': (60, 100),  # Example normal range for heart rate
        'SysBP': (90, 120),  # Example normal range for systolic blood pressure
        'RespRate': (12, 20),  # Example normal range for respiratory rate
        'SpO2': (95, 100)  # Example normal range for SpO2
    }

    # This is a list of your CSV filenames
    # csv_files = ['random_forest_model_predictions.csv',
    #              'ridge_model_predictions.csv',
    #              'xgboost_model_predictions.csv']
    csv_files = ['test.csv']
    for file in csv_files:
        print(file)
        # Load the CSV file
        df = pd.read_csv(file)

        # Check if required columns are present
        required_columns = ['HeartRate_pred', 'SysBP_pred', 'RespRate_pred', 'SpO2_pred']
        required_columns_2 = ['HeartRate', 'SysBP', 'RespRate', 'SpO2']
        if not all(col in df.columns for col in required_columns_2):
            print(f"One or more required columns are missing in {file}.")
            continue  # Skip processing for this file

        # Initialize severity score columns
        df['severity_score_1'] = 0  # Count of out-of-range measurements
        df['severity_score_2'] = 0  # Extreme measurement calculations
        df = calculate_severity(df, normal_ranges_2)
        check_severity_range(df)

        # Save the updated DataFrame back to CSV
        updated_file = f'updated_{file}'
        # df = df[
        #     ['subject_id', 'time_step', 'HeartRate_pred', 'SysBP_pred', 'RespRate_pred', 'SpO2_pred',
        #      'severity_score_1', 'severity_score_2']
        # ]
        print(df.columns)

        # 'subject_id', 'charttime', 'HeartRate', 'SysBP', 'RespRate', 'SpO2',
        # 'age', 'GENDER_F', 'GENDER_M', 'severity_score_1', 'severity_score_2',
        # 'HeartRate_extreme', 'HeartRate_severity_normalized', 'SysBP_extreme',
        # 'SysBP_severity_normalized', 'RespRate_extreme',
        # 'RespRate_severity_normalized', 'SpO2_extreme',
        # 'SpO2_severity_normalized'
        df = df[
            ['subject_id', 'charttime', 'HeartRate', 'SysBP', 'RespRate', 'SpO2',
             'severity_score_1', 'severity_score_2']
        ]
        df.to_csv(updated_file, index=False)


if __name__ == "__main__":
    main()