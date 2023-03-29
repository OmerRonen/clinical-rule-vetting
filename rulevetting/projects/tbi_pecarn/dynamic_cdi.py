import numpy as np
import pandas as pd

from imodels.tree.dfigs import D_FIGSClassifier
from imodels.tree.figs import FIGSClassifier

from rulevetting.projects.tbi_pecarn.dataset import Dataset as tbiDataset


def get_phases(columns):
    df_label = pd.read_csv('data/tbi_pecarn/TBI variables with label.csv')
    df_label = df_label.rename(
        columns={'Time (Aaron) 1= Prehospital, 2=primary survey, 3= first 1 hour, 4= > 1hour': 'timestamp'})
    time_distribution = df_label.groupby('timestamp')['Variable Name'].agg(list)

    phases = {0: [], 1: []}  # 0: prehospital, 1: hospital

    phase_1_features = time_distribution[0]
    phase_2_features = time_distribution[0] + time_distribution[2] + time_distribution[3]
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
    # phases[0] = list(set(phases[0]))
    # phases[0].sort()
    # phases[1] = list(set(phases[1]))
    # phases[1].sort()

    return phases


def main():
    data_train, data_tune, data_test = tbiDataset().get_data()
    phases = get_phases(data_train.columns)

    # add suffix for the features names indicating the phase
    def _get_phase(i):
        if i in phases[0]:
            return 1
        elif i in phases[1]:
            return 2
        else:
            return "NA"

    features_names = [f"{f}_phase_{_get_phase(i)}" for i, f in enumerate(data_train.drop(columns=['outcome']).columns)]
    d_figs = D_FIGSClassifier(phases=phases, max_rules=4)
    d_figs.feature_names_ = features_names
    figs = FIGSClassifier(max_rules=4)
    # drop nas in outcome and phase 1 variables
    data_train = data_train.dropna(subset=['outcome'] + [data_train.columns[i] for i in phases[0]])
    data_tune = data_tune.dropna(subset=['outcome'] + [data_tune.columns[i] for i in phases[0]])
    data_test = data_test.dropna(subset=['outcome'] + [data_test.columns[i] for i in phases[0]])
    d_figs_feature_names_ = [features_names[i] for i in phases[1]]

    d_figs.fit(data_train.drop(columns=['outcome']).values, data_train['outcome'].values, feature_names=d_figs_feature_names_)
    # fit figs on phase 1 variables
    figs_feature_names_ =  [features_names[i] for i in phases[0]]

    figs.fit(data_train.drop(columns=['outcome']).values[:, phases[0]], data_train['outcome'].values,feature_names=figs_feature_names_)

    print("==================D_FIGS===================")
    print(d_figs)
    print("==================FIGS===================")
    print(figs)

    # check that figs and d-figs predictions agree on phase 1
    figs_preds = figs.predict_proba(data_train.drop(columns=['outcome']).values[:, phases[0]])[:, 1]
    d_figs_preds = d_figs.predict_proba(data_train.drop(columns=['outcome']).values)[:, 1]

    # get phase 1, 2 indices - all rows where there are no nans in phase 2 variables
    phase_2_indices = data_train.drop(columns=['outcome']).iloc[:, phases[1]].notna().all(axis=1)
    phase_1_indices = np.invert(phase_2_indices)
    # check that prediction match on phase 1
    assert np.allclose(figs_preds[phase_1_indices], d_figs_preds[phase_1_indices])
    # plot two roc curves for two models phase two probas
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    fpr, tpr, _ = roc_curve(data_train['outcome'].values[phase_2_indices], figs_preds[phase_2_indices])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'FIGS (phase 1) (area = {roc_auc:.2f})')
    fpr, tpr, _ = roc_curve(data_train['outcome'].values[phase_2_indices], d_figs_preds[phase_2_indices])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'D_FIGS (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curves (phase 2)')
    ax.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    main()
