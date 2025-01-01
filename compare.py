import pandas as pd
import matplotlib.pyplot as plt

# Load healing logs
healing_log_fifo = pd.read_csv('sim2/healing_data_FIFO.csv')
healing_log_dynamic = pd.read_csv('sim2/healing_data.csv')
leaving_log_fifo = pd.read_csv('sim2/left_without_care_data_FIFO.csv')
leaving_log_dynamic = pd.read_csv('sim2/left_without_care_data.csv')





# Calculate number of patients and average severity scores for FIFO
num_patients_fifo = leaving_log_fifo['Patient ID'].nunique()
average_severity_score_1_fifo = leaving_log_fifo['Severity Score 1'].mean()
average_severity_score_2_fifo = leaving_log_fifo['Severity Score 2'].mean()

# Calculate number of patients and average severity scores for Dynamic
num_patients_dynamic = leaving_log_dynamic['Patient ID'].nunique()
average_severity_score_1_dynamic = leaving_log_dynamic['Severity Score 1'].mean()
average_severity_score_2_dynamic = leaving_log_dynamic['Severity Score 2'].mean()

# Print results
print(f"FIFO:\nNumber of Patients: {num_patients_fifo}\nAverage Severity Score 1: {average_severity_score_1_fifo:.4f}\nAverage Severity Score 2: {average_severity_score_2_fifo:.4f}\n")
print(f"Dynamic:\nNumber of Patients: {num_patients_dynamic}\nAverage Severity Score 1: {average_severity_score_1_dynamic:.4f}\nAverage Severity Score 2: {average_severity_score_2_dynamic:.4f}")





average_init_severity_fifo = healing_log_fifo.groupby('Start Care Time')['init severity_1'].mean().reset_index()
average_init_severity_fifo.columns = ['Time', 'Average Initial Severity FIFO']


# Calculate average initial severity for patients in care over time for Dynamic
average_init_severity_dynamic = healing_log_dynamic.groupby('Start Care Time')['init severity_1'].mean().reset_index()
average_init_severity_dynamic.columns = ['Time', 'Average Initial Severity Dynamic']




# Merge both results for comparison and fill NA with 0
comparison_df = pd.merge(average_init_severity_fifo, average_init_severity_dynamic, on='Time', how='outer')
comparison_df.fillna(0, inplace=True)  # Fill NA values with 0

comparison_df.sort_values(by='Time', inplace=True)
# Print the comparison table
print("\nAverage Initial Severity Over Time:")
print(comparison_df)

# Plotting the average initial severity
plt.figure(figsize=(12, 6))
plt.plot(comparison_df['Time'], comparison_df['Average Initial Severity FIFO'], label='FIFO Average Initial Severity', color='blue')
plt.plot(comparison_df['Time'], comparison_df['Average Initial Severity Dynamic'], label='Dynamic Average Initial Severity', color='orange')

plt.title('Average Initial Severity Over Time')
plt.xlabel('Time')
plt.ylabel('Average Initial Severity')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# Additional Comparisons
# Average waiting time before healing for FIFO
average_wait_time_fifo = healing_log_fifo['Departure Time'] - healing_log_fifo['Start Care Time']
average_wait_time_fifo = average_wait_time_fifo.mean()

# Average waiting time before healing for Dynamic
average_wait_time_dynamic = healing_log_dynamic['Departure Time'] - healing_log_dynamic['Start Care Time']
average_wait_time_dynamic = average_wait_time_dynamic.mean()

# Total number of patients treated in FIFO
total_patients_fifo = healing_log_fifo['Patient ID'].nunique()

# Total number of patients treated in Dynamic
total_patients_dynamic = healing_log_dynamic['Patient ID'].nunique()

# Average severity scores at departure
average_severity_score_fifo = healing_log_fifo[['Severity Score 1', 'Severity Score 2', 'actual_1', 'actual_2', 'init severity_1','init severity_2']].mean()
average_severity_score_dynamic = healing_log_dynamic[['Severity Score 1', 'Severity Score 2', 'actual_1', 'actual_2', 'init severity_1','init severity_2']].mean()

# Output the additional comparisons
print("\nAdditional Comparison Metrics:")
print(f"Average Waiting Time FIFO: {average_wait_time_fifo:.2f} time units")
print(f"Average Waiting Time Dynamic: {average_wait_time_dynamic:.2f} time units")
print(f"Total Patients Treated FIFO: {total_patients_fifo}")
print(f"Total Patients Treated Dynamic: {total_patients_dynamic}")
print(f"Average Departure Severity Score FIFO:\n{average_severity_score_fifo}")
print(f"Average Departure Severity Score Dynamic:\n{average_severity_score_dynamic}")
