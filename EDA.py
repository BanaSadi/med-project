import pandas as pd

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

# Sample DataFrame 'data' assumed to be already loaded with relevant data
# Calculate the count of males and females
gender_counts = final_data['GENDER'].value_counts()

# Calculate the total number of entries
total_entries = gender_counts.sum()

# Calculate the percentage of males and females
male_percentage = (gender_counts.get('M', 0) / total_entries) * 100
female_percentage = (gender_counts.get('F', 0) / total_entries) * 100

# Calculate the number of unique subject_ids
num_unique_subjects = final_data['subject_id'].nunique()

# Print the results
print(f"Percentage of Males: {male_percentage:.2f}%")
print(f"Percentage of Females: {female_percentage:.2f}%")
print(f"Number of unique subject_ids: {num_unique_subjects}")


# Calculate the number of unique subject_ids
unique_subjects = final_data['subject_id'].unique()

# Save the unique subject IDs to a text file
with open('unique_subjects.txt', 'w') as f:
    for subject_id in unique_subjects:
        f.write(f"{subject_id}\n")