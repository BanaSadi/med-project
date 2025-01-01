import numpy as np
import pandas as pd
from config import *

class Patient:
    def __init__(self, subject_id, arrival_time, severity_score_1, severity_score_2,
                 actual_severity_score_1, actual_severity_score_2):
        self.subject_id = subject_id
        self.arrival_time = arrival_time
        self.severity_score_1 = severity_score_1
        self.severity_score_2 = severity_score_2
        self.actual_severity_score_1 = actual_severity_score_1
        self.actual_severity_score_2 = actual_severity_score_2
        self.wait_time = 0
        self.status = 'waiting'  # 'waiting', 'in care', 'healed', 'left'
        self.intial_severity_1 = actual_severity_score_1
        self.intial_severity_2 = actual_severity_score_2
        self.start_care_time = None  # Time when care begins


def simulate_hospital_environment(subjects):
    waiting_room = []
    in_care = []
    healing_log = []
    left_without_care = []
    arrival_log = []

    unique_subject_ids = subjects[subjects['time_step'] == 1]['subject_id'].unique()

    for t in range(T):
        # Patient arrivals based on severity ranking
        arrivals = np.random.poisson(lambda_value)  # Poisson arrival at each timestep

        if unique_subject_ids.size > 0:
            num_patients_to_choose = min(arrivals, len(unique_subject_ids))
            print(f'At time {t}, {num_patients_to_choose} people arrived.')
            chosen_subject_ids = np.random.choice(unique_subject_ids, size=num_patients_to_choose, replace=False)

            for subject_id in chosen_subject_ids:
                row = subjects[(subjects['subject_id'] == subject_id) & (subjects['time_step'] == 1)].iloc[0]
                severity_score_1 = row['severity_score_1']
                severity_score_2 = row['severity_score_2']
                actual_score_1 = row['actual_severity_score_1']
                actual_score_2 = row['actual_severity_score_2']

                # Create new patient and add to waiting room
                new_patient = Patient(subject_id, t, severity_score_1, severity_score_2, actual_score_1, actual_score_2)
                waiting_room.append(new_patient)
                arrival_log.append((new_patient.subject_id, new_patient.arrival_time,
                                    new_patient.severity_score_1, new_patient.severity_score_2))

            # Remove chosen IDs from unique_subject_ids
            unique_subject_ids = np.setdiff1d(unique_subject_ids, chosen_subject_ids)

        else:
            print(f'at time {t} no patients arrived because no more exist in the system')


        # Process patients in care (handle treatment)
        if in_care:
            for patient in in_care.copy():
                if  patient.actual_severity_score_2 < 0.1:
                    patient.status = 'healed'
                    healing_log.append((patient.subject_id, patient.start_care_time, t, patient.intial_severity_1,
                                        patient.intial_severity_2,patient.severity_score_1, patient.severity_score_2,
                                        patient.actual_severity_score_1, patient.actual_severity_score_2))
                    in_care.remove(patient)


        # Move patients from waiting to care if there's available capacity
        if waiting_room and len(in_care) < M:  # Check if there's space in care
            # Sort waiting patients by severity
            waiting_room.sort(key=lambda x: (-x.severity_score_2, -x.severity_score_1, x.subject_id))  # Highest first

            # Fill care slots until capacity is reached
            patients_to_care_for = waiting_room[:M - len(in_care)]
            for patient in patients_to_care_for:
                patient.status = 'in care'
                patient.start_care_time = t
                in_care.append(patient)
                waiting_room.remove(patient)

        # Check for possible switches between patients in care and waiting
        while len(in_care) == M and waiting_room:  # Only check for switches if care is full
            # print("checking")
            # Identify the minimum severity patient in care and the maximum severity patient in waiting room
            lowest_severity_patient = min(in_care, key=lambda x: (x.severity_score_2, x.severity_score_1))
            highest_severity_patient = max(waiting_room, key=lambda x: (x.severity_score_2, x.severity_score_1))

            if (highest_severity_patient.severity_score_2 > lowest_severity_patient.severity_score_2):
                    # or(highest_severity_patient.severity_score_1 == lowest_severity_patient.severity_score_1 and
                    #  highest_severity_patient.severity_score_2 > lowest_severity_patient.severity_score_2)):

                # Perform the switch
                highest_severity_patient.status = 'in care'
                in_care.remove(lowest_severity_patient)
                in_care.append(highest_severity_patient)
                highest_severity_patient.start_care_time = t
                waiting_room.remove(highest_severity_patient)

                print(
                    f'Switched patient {lowest_severity_patient.subject_id} with patient {highest_severity_patient.subject_id} at time {t}.')
            else:
                # If no suitable switch can be made, exit the loop
                break

        # Update wait times for all waiting patients
        for patient in waiting_room:
            patient.wait_time += 1
            if patient.wait_time > W:  # If they waited too long, they leave
                patient.status = 'left'
                left_without_care.append((patient.subject_id, patient.arrival_time, t, patient.severity_score_1, patient.severity_score_2))
                waiting_room.remove(patient)

        for patient in waiting_room:
            subsequent_time_step = t - patient.arrival_time + 1
            # Get the severity scores for the patient pertaining to the current step
            row = subjects[(subjects['subject_id'] == patient.subject_id) & (subjects['time_step'] == subsequent_time_step)]
            if not row.empty:
                patient.severity_score_1 = row.iloc[0]['severity_score_1']
                patient.severity_score_2 = row.iloc[0]['severity_score_2']
                patient.actual_severity_score_1 = row.iloc[0]['actual_severity_score_1']
                patient.actual_severity_score_2 = row.iloc[0]['actual_severity_score_2']
                # print(row[['subject_id', 'severity_score_2', 'actual_severity_score_2']])

        # Update severity scores for patients in care
        for patient in in_care:
            subsequent_time_step = t - patient.arrival_time + 1
            # Get the severity scores for the patient pertaining to the current step
            row = subjects[(subjects['subject_id'] == patient.subject_id) & (subjects['time_step'] == subsequent_time_step)]
            if not row.empty:
                patient.severity_score_1 = row.iloc[0]['severity_score_1']
                patient.severity_score_2 = row.iloc[0]['severity_score_2']
                patient.actual_severity_score_1 = row.iloc[0]['actual_severity_score_1']
                patient.actual_severity_score_2 = row.iloc[0]['actual_severity_score_2']
                # print(row[['subject_id', 'severity_score_2',  'actual_severity_score_2']])

    return arrival_log, healing_log, left_without_care


def main():
    # Load and rank patients from the CSV file
    # Be sure to update the filename to match your actual CSV file
    subjects = pd.read_csv('predictions/updated_random_forest_model_predictions.csv')
    actuals = pd.read_csv('updated_test.csv')

    # Rename the column from 'severity_score_1' to 'actual_score_1'
    actuals.rename(columns={'severity_score_1': 'actual_severity_score_1', 'severity_score_2': 'actual_severity_score_2'}, inplace=True)

    actuals['time_step_'] = actuals.groupby('subject_id').cumcount()

    actuals['time_step_plus_1'] = actuals['time_step_'] + 1

    # Merge the DataFrames on subject_id and the condition that time_step in actuals is t-1 the one in subjects
    merged_df = pd.merge(subjects, actuals, how='inner', left_on=['subject_id', 'time_step'],
                         right_on=['subject_id', 'time_step_plus_1'])
    print(merged_df)
    arrival_log, healing_log, left_without_care = simulate_hospital_environment(merged_df)

    # Summarizing and displaying the results
    print("\nArrival Log (Patient ID, Arrival Time, Severity Score 1, Severity Score 2):")
    for log in arrival_log:
        print(log)

    print("\nHealing Log (Patient ID, Start Care Time, Departure Time):")
    for log in healing_log:
        print(log)

    print("\nLeft Without Care Log (Patient ID, Arrival Time, Departure Time):")
    for log in left_without_care:
        print(log)

    # Optionally, convert logs to Pandas DataFrames for easier manipulation if needed
    arrival_df = pd.DataFrame(arrival_log, columns=["Patient ID", "Arrival Time", "Severity Score 1", "Severity Score 2"])
    healing_df = pd.DataFrame(healing_log, columns=["Patient ID", "Start Care Time", "Departure Time",
                                                    'init severity_1', 'init severity_2', "Severity Score 1", "Severity Score 2", 'actual_1','actual_2'])
    left_df = pd.DataFrame(left_without_care, columns=["Patient ID", "Arrival Time", "Departure Time", "Severity Score 1", "Severity Score 2"])

    # Display DataFrames
    print("\nArrival DataFrame:")
    print(arrival_df)
    print("\nHealing DataFrame:")
    print(healing_df)
    print("\nLeft Without Care DataFrame:")
    print(left_df)

    arrival_df.to_csv('sim/arrival_data.csv', index=False)
    healing_df.to_csv('sim/healing_data.csv', index=False)
    left_df.to_csv('sim/left_without_care_data.csv', index=False)

if __name__ == "__main__":
    main()