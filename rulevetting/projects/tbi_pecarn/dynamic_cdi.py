import numpy as np
import pandas as pd


def get_phases(columns, permute_phases=False):
    df_label = pd.read_csv('data/tbi_pecarn/TBI variables with label.csv')
    df_label = df_label.rename(
        columns={'Time (Aaron) 1= Prehospital, 2=primary survey, 3= first 1 hour, 4= > 1hour': 'timestamp'})
    time_distribution = df_label.groupby('timestamp')['Variable Name'].agg(list)
    # df_label = pd.read_csv('results/dynamic_figs/tbi_pecarn/na_per_phase_annotated.csv')
    # time_distribution = df_label.groupby('Phase')['feature'].agg(list)
    time_distribution = [time_distribution[0], time_distribution[2], time_distribution[3]+time_distribution[4]]
    phases = {0: [], 1: [], 2: []}  # 0: prehospital, 1: hospital
    phase_1_idx = 0
    phase_2_idx = 1
    phase_3_idx = 2

    if permute_phases:
        phase_1_idx = 2
        phase_2_idx = 0
        phase_3_idx = 1


    phase_1_features = time_distribution[phase_1_idx]
    phase_2_features = time_distribution[phase_1_idx] + time_distribution[phase_2_idx]
    phase_3_features = time_distribution[phase_1_idx] + time_distribution[phase_2_idx] + time_distribution[phase_3_idx]

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
    phases[0] = np.array(list(set(phases[0])))
    phases[1] = np.array(list(set(phases[1])))
    phases[2] = np.array(list(set(phases[2])))

    # phases[0] = list(set(phases[0]))
    # phases[0].sort()
    # phases[1] = list(set(phases[1]))
    # phases[1].sort()

    return phases
