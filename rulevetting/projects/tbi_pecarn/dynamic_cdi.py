import numpy as np
import pandas as pd


def get_phases(columns):
    df_label = pd.read_csv('data/tbi_pecarn/TBI variables with label.csv')
    df_label = df_label.rename(
        columns={'Time (Aaron) 1= Prehospital, 2=primary survey, 3= first 1 hour, 4= > 1hour': 'timestamp'})
    time_distribution = df_label.groupby('timestamp')['Variable Name'].agg(list)

    phases = {0: [], 1: [], 2: []}  # 0: prehospital, 1: hospital

    phase_1_features = time_distribution[0]
    phase_2_features = time_distribution[0] + time_distribution[2]
    phase_3_features = time_distribution[0] + time_distribution[2] + time_distribution[3] + time_distribution[4]
    for feature in phase_1_features:
        # get all columns names that start with feature
        # set of new indices
        new_indices = [i for i, f in enumerate(columns) if f.startswith(feature)]
        phases[0] += new_indices
    # phases[1] = set(phases[0])
    for feature in phase_2_features:
        # get all columns names that start with feature

        new_indices = [i for i, f in enumerate(columns) if f.startswith(feature)]
        phases[1] += new_indices
    for feature in phase_3_features:
        # get all columns names that start with feature

        new_indices = [i for i, f in enumerate(columns) if f.startswith(feature)]
        phases[2] += new_indices

    # convert to numpy arrays
    phases[0] = np.array(phases[0])
    phases[1] = np.array(phases[1])
    phases[2] = np.array(phases[2])

    # phases[0] = list(set(phases[0]))
    # phases[0].sort()
    # phases[1] = list(set(phases[1]))
    # phases[1].sort()

    return phases