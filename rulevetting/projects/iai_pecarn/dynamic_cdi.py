import numpy as np
import pandas as pd

from rulevetting.projects.iai_pecarn.dataset import Dataset


def get_phases(columns, permute_phases=False):
    df_label = pd.read_csv('data/iai_pecarn/IAI variables with label.csv')
    df_label = df_label.rename(
        columns={'Time (Aaron) 1= Prehospital, 2=primary survey, 3= first 1 hour, 4= > 1hour': 'timestamp'})
    time_distribution = df_label.groupby('timestamp')['Variable Name'].agg(list)

    # join lists in time_distribution 1 and 2
    time_distribution.iloc[0] = time_distribution.iloc[1] + time_distribution.iloc[0]
    time_distribution.iloc[1] = time_distribution.iloc[2]
    phase_1_idx = 0
    phase_2_idx = 1
    if permute_phases:
        phase_1_idx = 1
        phase_2_idx = 0

    phases = {0: [], 1: []}

    phase_1_features = time_distribution.iloc[phase_1_idx] #phase 1
    phase_2_features = time_distribution.iloc[phase_1_idx] + time_distribution.iloc[phase_2_idx]

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

    phases[0] = np.array(list(set(phases[0])))
    phases[1] = np.array(list(set(phases[1])))


    return phases


def main():
    dset = Dataset()
    df_train, df_tune, df_test = dset.get_data(save_csvs=True, run_perturbations=False)
    phases = get_phases(df_train.columns)
    #run_sim(20, os.path.join("results", "dynamic_figs", "tbi_pecarn"))
    #plot_performance_results(os.path.join("results", "dynamic_figs", "tbi_pecarn"))

if __name__ == '__main__':
    main()
