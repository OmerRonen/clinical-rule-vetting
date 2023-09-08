import os
import copy
import pickle
import argparse
from typing import Tuple
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl

from imodels import FIGSClassifier, GreedyRuleListClassifier, RuleFitClassifier
from imodels.tree.dfigs import D_FIGSClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight



METHODS = [FIGSClassifier, RandomForestClassifier, GreedyRuleListClassifier, DecisionTreeClassifier,
           LogisticRegressionCV]
mpl.rcParams['font.size'] = 12       # Default font size for text elements
mpl.rcParams['axes.titlesize'] = 14  # Default font size for titles
mpl.rcParams['axes.labelsize'] = 12  # Default font size for axis labels
mpl.rcParams['xtick.labelsize'] = 10  # Default font size for x-axis tick labels
mpl.rcParams['ytick.labelsize'] = 10  # Default font size for y-axis tick labels
mpl.rcParams['legend.fontsize'] = 10  # Default font size for legends

def parse_args():
    # add dataset argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="csi_pecarn")
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


def plot_sensitivity_specificity_curve(preds, gt, ax=None):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    for method, probs in preds:
        fpr, tpr, _ = roc_curve(gt, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{method} (area = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')
    ax.legend(loc="lower right")
    # plt.show()


def get_aps(preds, gt):
    from sklearn.metrics import precision_recall_curve, auc
    aps = {}
    for method, probs in preds.items():
        precision, recall, _ = precision_recall_curve(gt, probs)
        auc_score = auc(recall, precision)
        aps[method] = auc_score
    return aps


def plot_prc(preds, gt, ax=None):
    from sklearn.metrics import precision_recall_curve, auc
    import matplotlib.pyplot as plt
    aps = {}
    if ax is None:
        fig, ax = plt.subplots()
    for method, probs in preds.items():
        precision, recall, _ = precision_recall_curve(gt, probs)
        auc_score = auc(recall, precision)
        aps[method] = auc_score
        ax.plot(recall[1:], precision[1:], label=f"{method} (area = {auc_score:.2f})")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    return aps


def plot_rocs(preds, gt, ax=None):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    aucs = {}
    if ax is None:
        fig, ax = plt.subplots()
    for method, probs in preds.items():
        fpr, tpr, _ = roc_curve(gt, probs)
        roc_auc = auc(fpr, tpr)
        aucs[method] = roc_auc
        ax.plot(fpr, tpr, label=f"{method} (area = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    return aucs
    # ax.set_title('ROC curve')
    # ax.legend(loc="lower right")
    # plt.show()


def get_aucs(preds, gt):
    from sklearn.metrics import roc_curve, auc
    aucs = {}

    for method, probs in preds.items():
        fpr, tpr, _ = roc_curve(gt, probs)
        roc_auc = auc(fpr, tpr)
        aucs[method] = roc_auc

    return aucs


def get_phase_indices(data, phase_features):
    idx = data.iloc[:, phase_features].notna().all(axis=1)
    return idx


def _get_phase(i, phases):
    for phase in phases:
        if i in phases[phase]:
            return phase + 1

    return "NA"


def fit_methods(X_train, X_train_imputed, y_train, phases, methods):
    phases['imputed'] = phases[len(phases) - 1]
    method_per_phase = {phase: {} for phase in phases}
    for phase, vars in phases.items():
        if phase != "imputed":
            idx = get_phase_indices(X_train, vars)
            X_train_phase = X_train.iloc[:, vars]
        else:
            idx = get_phase_indices(X_train_imputed, vars)
            X_train_phase = X_train_imputed.iloc[:, vars]

        for method in methods:
            cls = method()
            cls.fit(X_train_phase.loc[idx, :].values, y_train[idx].values)
            method_per_phase[phase][method.__name__] = cls
    return method_per_phase


def plot_predictions(X_test, y_test, d_figs, method, phases, phase):
    phase_vars = phases[phase]
    phase_idx = get_phase_indices(X_test, phase_vars)
    if phase + 1 in phases:
        phase_idx = np.logical_and(phase_idx, np.invert(get_phase_indices(X_test, phases[phase + 1])))
    y_test = y_test[phase_idx]
    X_test_phase = X_test.iloc[:, phase_vars]
    preds_dfigs = d_figs.predict_proba(X_test.loc[phase_idx, :].values)[:, 1]
    preds_method = method.predict_proba(X_test_phase.loc[phase_idx, :].values)[:, 1]
    # do a scatter plot for the predictions and highlight the true labels by making it bigger and red
    fig, ax = plt.subplots()
    # plot prediction on true cases and make red and large

    idx_true = y_test == 1
    ax.scatter(preds_dfigs[idx_true], preds_method[idx_true], c='r', s=20 * y_test[idx_true] + 1, label="Tbi")
    # now predictions on false cases
    idx_false = y_test == 0
    ax.scatter(preds_dfigs[idx_false], preds_method[idx_false], c='b', s=20 * y_test[idx_false] + 1, label="No Tbi")
    ax.set_xlabel(d_figs.__class__.__name__)
    method_name = method.__class__.__name__
    ax.set_ylabel(method_name)
    ax.set_title(f"Phase {phase + 1}", fontsize=20)
    # make it figure out the range according to the data
    min_pred = np.min([preds_dfigs, preds_method])
    max_pred = np.max([preds_dfigs, preds_method])
    ax.set_xlim([min_pred, max_pred])
    ax.set_ylim([min_pred, max_pred])
    ax.plot([min_pred, max_pred], [min_pred, max_pred], 'k--')
    ax.legend()
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 1])

    # ax.set_title(f"Phase {phase}")


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


def get_phase(feature, data):
    phases = get_phases(data.columns)
    for phase in phases:
        if feature in phases[phase]:
            return phase + 1

    return None


def save_na_per_phase(results_dir):
    data_train, data_tune, data_test = Dataset().get_data()
    # join three splits
    data = pd.concat([data_train, data_tune, data_test])
    output = {"feature": [], "phase": [], "pct_na": []}
    for i, feature in enumerate(data.columns):
        if feature.split("_")[0] in output["feature"]:
            continue
        phase = get_phase(i, data)
        if phase is None:
            continue
        output["feature"].append(feature.split("_")[0])
        # output["feature"].append(feature)
        output["phase"].append(phase)
        output["pct_na"].append(data[feature].isna().mean())
    # save to csv
    df = pd.DataFrame(output)
    # sort df by phase
    df = df.sort_values(by="pct_na")
    df.to_csv(os.path.join(results_dir, "na_per_phase.csv"), index=False)


def log_phases(X, permute_phases):
    phases = get_phases(X.columns, permute_phases)
    for phase in phases:
        print(f"phase {phase} shape: {get_phase_indices(X, phases[phase]).sum()}")


def add_na_dummy(df):
    df_cpy = df.copy()
    for column in df_cpy.columns:
        if df_cpy[column].isna().any():
            na_column = f'NaN_{column}'
            df_cpy[na_column] = df[column].isna().astype(int)
            df_cpy[column].fillna(0, inplace=True)
    return df_cpy


def run_sim(n_seeds, results_dir, max_rules=5, permute_phase=False, max_trees = None):
    X_imputed, y_imputed = get_dataset(impute=True, permute_phase=permute_phase)
    X, y = get_dataset(impute=False, permute_phase=permute_phase)
    if max_trees == 1:
        method = "CART"
    else:
        method = "FIGS"
    #method = "CART" if max_trees == 1 else "FIGS"
    # remove duplicates columns in X and X_imputed
    # X = X.loc[:, ~X.columns.duplicated()]
    # X_imputed = X_imputed.loc[:, ~X_imputed.columns.duplicated()]
    # create a results dir direcotry if it does not exist
    results_dir = os.path.join(results_dir, method.lower())
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    log_phases(X, permute_phase)
    phases = get_phases(X.columns, permute_phases=permute_phase)
    performance = {"dfigs": [], "dfigs_imputed": [], "figs_na": [], "figs_imputed": [], "figs": []}
    performance_per_phase = {"dfigs" : {i:[] for i in range(len(phases))},
                             "dfigs_imputed": {i:[] for i in range(len(phases))},
                             "figs_na": {i:[] for i in range(len(phases))},
                            "figs_refit": {i:[] for i in range(len(phases))},
                            "figs_refit_imputed": {i:[] for i in range(len(phases))
                            }}
    for seed in range(n_seeds):
        X_na = add_na_dummy(X)

        # get train and test indices
        train_idx, test_idx = train_test_split(np.arange(X.shape[0]), test_size=0.4, random_state=seed)
        X_train, X_test, y_train, y_test = X.iloc[train_idx, :], X.iloc[test_idx, :], y.iloc[train_idx], y.iloc[test_idx]
        X_train_imputed, X_test_imputed = X_imputed.iloc[train_idx, :], X_imputed.iloc[test_idx, :]
        
        X_na_train, X_na_test = X_na.iloc[train_idx, :], X_na.iloc[test_idx, :]

        # set idx to be all the rows without any na values in X_test. Have information for all three phases. 
        idx = X_test.dropna().index

        
        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
    

        d_figs = D_FIGSClassifier(phases=copy.deepcopy(phases), max_rules=max_rules)
        d_figs_auc = get_auc_score(d_figs, X_train, X_test, y_train, y_test, idx=idx,sample_weight = sample_weight)

        d_figs_imputed = D_FIGSClassifier(phases=copy.deepcopy(phases), max_rules=max_rules)
        d_figs_imputed_auc = get_auc_score(d_figs, X_train_imputed, X_test_imputed, y_train, y_test, idx=idx,sample_weight = sample_weight)
        
        figs_na = FIGSClassifier(max_rules=max_rules * len(phases), max_trees=max_trees)
        figs_imp = FIGSClassifier(max_rules=max_rules * len(phases), max_trees=max_trees)
        figs = FIGSClassifier(max_rules=max_rules * len(phases),   max_trees=max_trees)

        figs_na_auc = get_auc_score(figs_na, X_na_train, X_na_test, y_train, y_test, idx=idx,sample_weight = sample_weight)
        figs_imp_auc = get_auc_score(figs_imp, X_train_imputed, X_test_imputed, y_train, y_test, idx=idx,sample_weight = sample_weight)

        X_train_no_na, y_train_no_na = X_train.dropna(), y_train[X_train.dropna().index]
        sample_weight_no_na = compute_sample_weight(class_weight = 'balanced', y = y_train_no_na)
        figs_auc = get_auc_score(figs, X_train_no_na, X_test, y_train_no_na, y_test, idx=idx,sample_weight = sample_weight_no_na)
        
        # add this to performance dict
        performance["dfigs"].append(d_figs_auc)
        performance["dfigs_imputed"].append(d_figs_imputed_auc)
        performance["figs_na"].append(figs_na_auc)
        performance["figs_imputed"].append(figs_imp_auc)
        performance["figs"].append(figs_auc)

        #
        phases_idx_test = get_phase_idx(X_test, phases)

        for phase, idx, in phases_idx_test.items():
            if np.var(y_test[idx]) == 0:
                continue
            print(f"phase: {phase}")
            figs_refit = FIGSClassifier(max_rules = max_rules * (phase + 1), max_trees = max_trees)
            
            X_train_phase = copy.deepcopy(X_train).iloc[:,phases[phase]].dropna()
            X_train_phase_imputed = copy.deepcopy(X_train_imputed).iloc[:,phases[phase]]

            print(f"X_train shape: {X_train_phase.shape}")
            
            X_test_phase =  copy.deepcopy(X_test).iloc[:,phases[phase]]
            X_test_phase_imputed =  copy.deepcopy(X_test_imputed).iloc[:,phases[phase]]
            
            y_train_phase = y_train[X_train.iloc[:,phases[phase]].dropna().index]
            sample_weight_phase = compute_sample_weight(class_weight='balanced', y=y_train_phase)
            sample_weight_phase_imputed = compute_sample_weight(class_weight='balanced', y=y_train)
            
            performance_per_phase["figs_refit"][phase].append(get_auc_score(figs_refit,X_train_phase,X_test_phase,y_train_phase,y_test,idx = idx, sample_weight = sample_weight_phase))
            #performance_per_phase["figs_refit_imputed"][phase].append(get_auc_score(figs_refit,X_train_phase_imputed,X_test_phase_imputed,y_train_phase,y_test,idx = idx, sample_weight = sample_weight_phase))
            performance_per_phase["dfigs"][phase].append(get_auc_score(d_figs, X_train, X_test, y_train, y_test, idx=idx,sample_weight = sample_weight))
            performance_per_phase["dfigs_imputed"][phase].append(get_auc_score(d_figs_imputed, X_train_imputed, X_test_imputed, y_train, y_test, idx=idx,sample_weight = sample_weight))
            performance_per_phase["figs_na"][phase].append(get_auc_score(figs_na, X_na_train, X_na_test, y_train, y_test, idx=idx,sample_weight = sample_weight))

        print(performance_per_phase)


    fig, axs = plt.subplots(1, len(phases), figsize=(15, 15))
    max_auc = -1 * np.inf
    min_auc = np.inf
    for i, phase in phases.items():
        # do two bar plots one next to the other for dfigs and figs_na
        axs[i].bar([f"D-{method}",f"D-{method} (imputed)", f"{method} (na category)",f"{method} (refit)", ], [np.mean(performance_per_phase["dfigs"][i]),np.mean(performance_per_phase["dfigs_imputed"][i]), np.mean(performance_per_phase["figs_na"][i]),np.mean(performance_per_phase["figs_refit"][i])],
        yerr=[np.std(performance_per_phase["dfigs"][i]) / np.sqrt(n_seeds), np.std(performance_per_phase["dfigs_imputed"][i]) / np.sqrt(n_seeds), np.std(performance_per_phase["figs_na"][i]) / np.sqrt(n_seeds), np.std(performance_per_phase["figs_refit"][i]) / np.sqrt(n_seeds)],
                     color=["#1f77b4", "red", "#ff7f0e","green"])
        axs[i].set_title(f"phase {i}")
        # update max and min auc
        max_std_phase = 3 * np.max([np.std(performance_per_phase["dfigs"][i]) / np.sqrt(n_seeds), np.std(performance_per_phase["dfigs_imputed"][i]) / np.sqrt(n_seeds), np.std(performance_per_phase["figs_na"][i]) / np.sqrt(n_seeds), np.std(performance_per_phase["figs_refit"][i]) / np.sqrt(n_seeds)])
        max_auc_phase = np.max([np.mean(performance_per_phase["dfigs"][i]), np.mean(performance_per_phase["dfigs_imputed"][i]), np.mean(performance_per_phase["figs_na"][i]),np.mean(performance_per_phase["figs_refit"][i])])
        min_auc_phase = np.min([np.mean(performance_per_phase["dfigs"][i]), np.mean(performance_per_phase["dfigs_imputed"][i]), np.mean(performance_per_phase["figs_na"][i]),np.mean(performance_per_phase["figs_refit"][i])])
        max_auc = max(max_auc, max_auc_phase + max_std_phase)
        min_auc = min(min_auc, min_auc_phase - max_std_phase)
    for i in range(len(phases)):
        axs[i].set_ylim(min_auc, max_auc)


    # tight layout
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "bar_plot_figs_per_phase.png"), dpi=300)
    plt.close()
    # make a bar plot for the three methods with error bars
    fig, ax = plt.subplots()
    #method = "FIGS" if max_trees is None else "CART"
    methods_names = [f"D-{method}",f"D-{method} (imputed)", f"{method} (na category)", f"{method} (imputed)", method]
    methods_average_auc = [np.mean(performance["dfigs"]), np.mean(performance["dfigs_imputed"]), np.mean(performance["figs_na"]),
                           np.mean(performance["figs_imputed"]),
                           np.mean(performance["figs"])]
    methods_std_auc = [np.std(performance["dfigs"]) / np.sqrt(n_seeds),
                       np.std(performance["dfigs_imputed"]) / np.sqrt(n_seeds),
                       np.std(performance["figs_na"]) / np.sqrt(n_seeds),
                       np.std(performance["figs_imputed"]) / np.sqrt(n_seeds),
                       np.std(performance["figs"]) / np.sqrt(n_seeds)]
    ax.bar(methods_names, methods_average_auc, yerr=methods_std_auc, align='center', alpha=0.5, ecolor='black',
           capsize=8)
    # make x-ticks smaller
    # ax.set_xticks(fontsize=10)
    # set the y-axis limits
    ax.set_ylim(0.5, 1)
    ax.set_ylabel('AUC')
    ttl = "AUC per method on last phase patients (permuted phases)" if permute_phase else "AUC per method on last phase patients"
    ax.set_title(ttl)
    plt.tight_layout()
    # save the figure
    plt.savefig(os.path.join(results_dir, "bar_plot_figs.png"), dpi=300)
    plt.close()

    # d_figs.feature_names_ = features_names
    # d_figs.fit(X_train.values, y_train.values)
    # phases_idx = get_phase_idx(X_train, phases)
    # method_per_phase = fit_methods(X_train, X_train_imputed, y_train, copy.deepcopy(phases), METHODS)
    # for cdi_strategy in ["current", "first", "imputed"]:
    #     scores_strategy = get_scores(cdi_strategy, phases_idx, method_per_phase, d_figs, X_test, X_test_imputed,
    #                                  y_test, permute_phase)
    #     for phase in scores_strategy["auroc"]:
    #
    #         for m in scores_strategy["auroc"][phase]:
    #             results[cdi_strategy][phase]["auc"][m].append(scores_strategy["auroc"][phase][m])

    # def plot_curves(curve_type, analysis_type):
    #     fig, axs = plt.subplots(1, 3, figsize=(15, 15))
    #     plt_func = plot_rocs if curve_type == "auc" else plot_prc
    #     phase_methods_ = 0 if analysis_type == "first" else None
    #     title = f"{curve_type.upper()} for D_FIGS vs different methods trained on {analysis_type} phase variables"
    #     fig_name = f"{analysis_type}_{curve_type}.png"
    #     for phase, phase_vars in phases.items():
    #         phase_methods = method_per_phase[phase] if phase_methods_ is None else method_per_phase[0]
    #         phase_vars = phase_vars if phase_methods_ is None else phases[0]
    #         phase_idx = get_phase_indices(X_test, phase_vars)
    #         if phase + 1 in phases:
    #             phase_idx = np.logical_and(phase_idx, np.invert(get_phase_indices(X_test, phases[phase + 1])))
    #         X_test_phase = X_test.iloc[:, phase_vars]
    #         preds = {m: clf.predict_proba(X_test_phase.loc[phase_idx, :].values)[:, 1] for m, clf in
    #                  phase_methods.items()}
    #         if phase > 0:
    #             preds[d_figs.__class__.__name__] = d_figs.predict_proba(X_test.loc[phase_idx, :].values)[:, 1]
    #         plt_func(preds, y_test[phase_idx], ax=axs[phase])
    #         axs[phase].set_title(f"Phase {phase} (# of patients: {phase_idx.sum()})")
    #         loc = "lower right" if curve_type == "auc" else "upper right"
    #         axs[phase].legend(loc=loc)
    #     plt.suptitle(title)
    #     plt.savefig(f"{results_dir}/{fig_name}")
    #
    # if seed == 1:
    #     plot_curves("auc", "first")
    #     plot_curves("auc", "current")
    #     plot_curves("auprc", "first")
    #     plot_curves("auprc", "current")
    #
    #     plot_predictions(X_test, y_test, d_figs, method_per_phase[0]['FIGSClassifier'], phases, phase=0)
    #     plt.savefig(os.path.join(results_dir, "phase_1_predictions.png"))
    #     plt.close()
    #     plot_predictions(X_test, y_test, d_figs, method_per_phase[1]["FIGSClassifier"], phases, phase=1)
    #     plt.savefig(os.path.join(results_dir, "phase_2_predictions.png"))
    #     plt.close()
    #     try:
    #         plot_predictions(X_test, y_test, d_figs, method_per_phase[2]["FIGSClassifier"], phases, phase=2)
    #         plt.savefig(os.path.join(results_dir, "phase_2_predictions.png"))
    #         plt.close()
    #     except KeyError:
    #         continue

    # save aucs to pickle
    with open(f"{results_dir}/performance.pkl", "wb") as f:
        pickle.dump(performance, f)


def plot_performance_results(results_dir):
    # load results from pickle
    with open(f"{results_dir}/results.pkl", "rb") as f:
        results = pickle.load(f)

    # make bar plots for auc and aucpr for each phase for current key in results
    def save_bar_plot(score_type):
        for key, scores in results.items():
            fig, axs = plt.subplots(1, 3, figsize=(15, 15))
            min_y = 1
            max_y = 0
            for phase, phase_scores in scores.items():
                # fig, ax = plt.subplots(1, 1, figsize=(15, 15))
                for method, scores in phase_scores[score_type].items():
                    std_mean = 2 * np.std(scores) / np.sqrt(len(scores))
                    axs[phase].bar(method, np.mean(scores), yerr=std_mean, label=method)
                    min_y = min(min_y, np.mean(scores) - std_mean)
                    max_y = max(max_y, np.mean(scores) + std_mean)

                axs[phase].set_title(f"phase {phase + 1}", fontsize=20)
                if phase == 0:
                    axs[phase].legend(loc="upper right", fontsize=14)
                # remove xticks
                axs[phase].set_xticks([])
                # make y ticks size larger
                axs[phase].tick_params(axis='y', labelsize=20)
            # set the same y limits for all plots
            for ax in axs:
                # ax.set_ylim(min_y, max_y + 0.05)
                ax.set_ylim(0.2, 1)

            # k = "oracle" if key == "current" else "phase_1"
            ttl = f"cdi strategy - {key} phase"
            plt.suptitle(ttl, fontsize=20)
            plt.tight_layout()
            plt.savefig(f"{results_dir}/{key}_{score_type}_bar.png")

            plt.close()

    save_bar_plot("auc")
    # save_bar_plot("auprc")
    # for key, scores in results.items():
    #     for phase, phase_scores in scores.items():
    #         for score_type, score_dict in phase_scores.items():
    #             fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    #             for method, scores in score_dict.items():
    #                 ax.bar(method, np.mean(scores), yerr=np.std(scores), label=method)
    #             ax.set_title(f"{score_type.upper()} for {key} phase variables")
    #             ax.legend()
    #             plt.savefig(f"results/dynamic_figs/{key}_{phase}_{score_type}.png")
    #             plt.close()


def get_phase_idx(X_test, phases):
    # unique observations for each phase
    phases_idx = {}
    for phase in phases:
        phase_idx = get_phase_indices(X_test, phases[phase])
        if phase + 1 in phases:
            phase_idx = np.logical_and(phase_idx, np.invert(get_phase_indices(X_test, phases[phase + 1])))
        phases_idx[phase] = phase_idx
    return phases_idx


def get_scores(cdi_strategy: str, phases_idx: dict, fitted_methods: dict, d_figs: D_FIGSClassifier,
               X_test: pd.DataFrame,
               X_test_imputed: pd.DataFrame,
               y_test: pd.Series,
               permute_phases: bool) -> dict:
    """AUC and AUPRC for each phase for each method

    Args:
        cdi_strategy (str): strategy to derive CDI we are comparing against:
            current - different model for each phase
            first - model fitted using phase 1 features
            imputed - model fitted using imputed data)
        phases_idx (dict): unique observations for each phase
        fitted_methods (dict): fitted methods for each phase
        d_figs (D_FIGSClassifier): fitted dynamic FIGS classifier
        X_test (pd.DataFrame): test data
        y_test (pd.Series): test labels
        permute_phases (bool): whether to permute phases

    Returns:
        dict: AUC and AUPRC for each phase for each method

    """
    phases = get_phases(X_test.columns, permute_phases)
    phases['imputed'] = phases[len(phases) - 1]
    aucs_phase = {}
    aucprs_phase = {}
    for phase, phase_idx in phases_idx.items():
        if cdi_strategy == "current":
            methods_phase = phase
        elif cdi_strategy == "first":
            methods_phase = 0
        elif cdi_strategy == "imputed":
            methods_phase = "imputed"
        else:
            raise ValueError("Invalid cdi strategy")
        X_test_phase = X_test.iloc[:, phases[methods_phase]] if cdi_strategy != "imputed" else X_test_imputed.iloc[:,
                                                                                               phases[methods_phase]]
        preds = {name: method.predict_proba(X_test_phase.loc[phase_idx, :].values)[:, 1] for name, method in
                 fitted_methods[methods_phase].items()}
        d_figs_preds = d_figs.predict_proba(X_test.loc[phase_idx, :].values)[:, 1]
        preds["D_FIGSClassifier"] = d_figs_preds

        aucs_phase[phase] = get_aucs(preds, y_test[phase_idx])
        aucprs_phase[phase] = get_aps(preds, y_test[phase_idx])

    return {"auroc": aucs_phase, "auprc": aucprs_phase}


def get_auc_score(cls, X_train, X_test, y_train, y_test, idx=None,sample_weight = None):
    print(X_train)
    if isinstance(cls,D_FIGSClassifier):
        cls.fit(X_train.values, y_train,use_class_weight = True)
    else:
        cls.fit(X_train.values.astype(float), y_train,sample_weight = sample_weight) #values
    if idx is not None:
        preds = cls.predict_proba(X_test.loc[idx, :].values.astype(float))[:, 1]
        if len(np.unique(preds)) == 1:
            return 0.5
        if np.var(y_test[idx]) == 0:
            return np.nan
        return roc_auc_score(y_test[idx], preds)
    # get phase of first split
    return roc_auc_score(y_test, cls.predict_proba(X_test.values.astype(float))[:, 1])


def main():
    n_seeds = 10
    results_dir = os.path.join("results", "dynamic_methods", DS)
    run_sim(n_seeds, results_dir, permute_phase=False,max_trees=None)
    run_sim(n_seeds, results_dir, permute_phase=False, max_trees=1)

    # plot_performance_results(results_dir)

    # results_dir = os.path.join("results", "dynamic_figs_permuted", DS)
    # run_sim(n_seeds, results_dir, permute_phase=True)
    # plot_performance_results(results_dir)


if __name__ == '__main__':
    main()
