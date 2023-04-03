import argparse
import copy
import os.path
from typing import Tuple

import numpy as np
import pandas as pd
from imodels import FIGSClassifier
from imodels.tree.dfigs import D_FIGSClassifier

from rulevetting.projects.dynamic_cdi_utils import get_phase_idx, get_phase_indices
from sklearn.model_selection import train_test_split

def parse_args():
    # add dataset argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="iai_pecarn")
    args = parser.parse_args()
    return args


args = parse_args()
DS = args.dataset

if DS == "tbi_pecarn":
    from rulevetting.projects.tbi_pecarn.dynamic_cdi import get_phases
    from rulevetting.projects.tbi_pecarn.dataset import Dataset
elif DS == "csi_pecarn":
    from rulevetting.projects.csi_pecarn.dynamic_cdi import get_phases
    from rulevetting.projects.csi_pecarn.dataset1 import Dataset
elif DS == "iai_pecarn":
    from rulevetting.projects.iai_pecarn.dynamic_cdi import get_phases
    from rulevetting.projects.iai_pecarn.dataset import Dataset

def get_dataset(impute: bool, permute_phase: bool) -> Tuple[pd.DataFrame, pd.Series]:
    """Get processed tbi data

    Args:
        impute (bool): if true missing values are imputed with mode
        permute_phase (bool): if true the phases are permutred

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X, y

    """
    data_train, data_tune, data_test = Dataset().get_data()
    # join three splits
    data = pd.concat([data_train, data_tune, data_test])
    # separate outcome
    y = data['outcome']
    # drop outcome
    X = data.drop(columns=['outcome'])
    phases = get_phases(X.columns, permute_phases=permute_phase)

    last_phase = np.array(phases[np.max(list(phases.keys()))])
    # select last_phase columns in X
    X = X.iloc[:, last_phase]
    phases = get_phases(X.columns, permute_phases=permute_phase)
    idx_first = get_phase_indices(X, phases[np.min(list(phases.keys()))])
    X, y = X.loc[idx_first, :], y[idx_first]
    # now impute missing values with mode
    if impute:
        X = X.fillna(X.mode().iloc[0])
    return X, y

def main():
    results_dir = os.path.join("results", "backcompat", DS)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    X, y = get_dataset(impute=False, permute_phase=False)
    phases = get_phases(X.columns, permute_phases=False)
    d_figs = D_FIGSClassifier(phases=copy.deepcopy(phases), max_rules=2, max_trees=1)

    # make train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # fit model
    d_figs.fit(X_train.values, y_train)
    # plot the model and save it
    d_figs.plot(dpi=300, filename=os.path.join(results_dir, "dfigs.png"), feature_names=X.columns)

    phases_idx_test = get_phase_idx(X_test, phases)
    for phase, idx, in phases_idx_test.items():
        if np.var(y_test[idx]) == 0:
            continue
        X_train_phase, X_test_phase = X_train.iloc[:, phases[phase]], X_test.iloc[:, phases[phase]]
        na_idx_train, na_idx_test = X_train_phase.dropna().index, X_test_phase.dropna().index
        X_train_phase, X_test_phase = X_train_phase.loc[na_idx_train, :], X_test_phase.loc[na_idx_test, :]
        y_train_phase, y_test_phase = y_train[na_idx_train], y_test[na_idx_test]
        figs = FIGSClassifier(max_rules=6, max_trees=1)
        figs.fit(X_train_phase.values, y_train_phase)
        figs.plot(dpi=300, filename=os.path.join(results_dir, f"figs_{phase}.png"), feature_names=X_train_phase.columns)


if __name__ == '__main__':
    main()

