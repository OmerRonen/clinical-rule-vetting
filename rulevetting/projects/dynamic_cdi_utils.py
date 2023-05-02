import os
import copy
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from imodels import FIGSClassifier, GreedyRuleListClassifier, RuleFitClassifier
from imodels.tree.dfigs import D_FIGSClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# from rulevetting.projects.csi_pecarn.dynamic_cdi import get_phases
# from rulevetting.projects.csi_pecarn.dataset1 import Dataset

from rulevetting.projects.tbi_pecarn.dynamic_cdi import get_phases
from rulevetting.projects.tbi_pecarn.dataset import Dataset

METHODS = [FIGSClassifier, RandomForestClassifier, GreedyRuleListClassifier, LogisticRegressionCV]


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


def get_dataset(impute: bool) -> Tuple[pd.DataFrame, pd.Series]:
    """Get processed tbi data

    Args:
        impute (bool): if true missing values are imputed with mode

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
    phases = get_phases(X.columns)

    last_phase = np.array(phases[np.max(list(phases.keys()))])
    # select last_phase columns in X
    X = X.iloc[:, last_phase]
    phases = get_phases(X.columns)
    idx_first = get_phase_indices(X, phases[np.min(list(phases.keys()))])
    X, y = X.loc[idx_first, :], y[idx_first]
    # now impute missing values with mode
    if impute:
        X = X.fillna(X.mode().iloc[0])
    return X, y


def run_sim(n_seeds, results_dir, max_rules=5):
    X_imputed, y_imputed = get_dataset(impute=True)
    X, y = get_dataset(impute=False)

    phases = get_phases(X.columns)
    for phase in phases:
        print(f"phase {phase} shape: {get_phase_indices(X, phases[phase]).sum()}")

    features_names = [f"{f}_phase_{_get_phase(i, phases)}" for i, f in enumerate(X.columns)]
    methods_dict = {m.__name__: [] for m in METHODS + [D_FIGSClassifier]}
    results_template = {phase: {"auc": copy.deepcopy(methods_dict), "auprc": copy.deepcopy(methods_dict)} for phase in
                        phases}
    results = {"current": copy.deepcopy(results_template), "first": copy.deepcopy(results_template),
               "imputed": copy.deepcopy(results_template)}
    for seed in range(n_seeds):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)
        X_train_imputed, X_test_imputed, y_train_imputed, y_test_imputed = train_test_split(X_imputed, y_imputed,
                                                                                            test_size=0.4,
                                                                                            random_state=seed)

        phases_idx = get_phase_idx(X_test, phases)

        assert np.sum(y_test) == np.sum(y_test_imputed)

        d_figs = D_FIGSClassifier(phases=copy.deepcopy(phases), max_rules=max_rules)
        d_figs.feature_names_ = features_names
        d_figs.fit(X_train.values, y_train.values)

        method_per_phase = fit_methods(X_train, X_train_imputed, y_train, copy.deepcopy(phases), METHODS)
        for cdi_strategy in ["current", "first", "imputed"]:
            scores_strategy = get_scores(cdi_strategy, phases_idx, method_per_phase, d_figs, X_test, X_test_imputed,
                                         y_test)
            # scores_seed_first = get_scores("first", phases, method_per_phase, d_figs, X_test, y_test)
            # scores_seed_current = get_scores("current", phases, method_per_phase, d_figs, X_test, y_test)
            # #
            for phase in scores_strategy["auroc"]:

                for m in scores_strategy["auroc"][phase]:
                    results[cdi_strategy][phase]["auc"][m].append(scores_strategy["auroc"][phase][m])
                    # results["current"][phase]["auc"][m].append(scores_seed_current["auroc"][phase][m])
                    # results["current"][phase]["auprc"][m].append(scores_seed_current["auprc"][phase][m])
                    #
                    # results["first"][phase]["auc"][m].append(scores_seed_first["auroc"][phase][m])
                    # results["first"][phase]["auprc"][m].append(scores_seed_first["auprc"][phase][m])

        def plot_curves(curve_type, analysis_type):
            fig, axs = plt.subplots(1, 3, figsize=(15, 15))
            plt_func = plot_rocs if curve_type == "auc" else plot_prc
            phase_methods_ = 0 if analysis_type == "first" else None
            title = f"{curve_type.upper()} for D_FIGS vs different methods trained on {analysis_type} phase variables"
            fig_name = f"{analysis_type}_{curve_type}.png"
            for phase, phase_vars in phases.items():
                phase_methods = method_per_phase[phase] if phase_methods_ is None else method_per_phase[0]
                phase_vars = phase_vars if phase_methods_ is None else phases[0]
                phase_idx = get_phase_indices(X_test, phase_vars)
                if phase + 1 in phases:
                    phase_idx = np.logical_and(phase_idx, np.invert(get_phase_indices(X_test, phases[phase + 1])))
                X_test_phase = X_test.iloc[:, phase_vars]
                preds = {m: clf.predict_proba(X_test_phase.loc[phase_idx, :].values)[:, 1] for m, clf in
                         phase_methods.items()}
                if phase > 0:
                    preds[d_figs.__class__.__name__] = d_figs.predict_proba(X_test.loc[phase_idx, :].values)[:, 1]
                plt_func(preds, y_test[phase_idx], ax=axs[phase])
                axs[phase].set_title(f"Phase {phase} (# of patients: {phase_idx.sum()})")
                loc = "lower right" if curve_type == "auc" else "upper right"
                axs[phase].legend(loc=loc)
            plt.suptitle(title)
            plt.savefig(f"{results_dir}/{fig_name}")

        if seed == 1:
            plot_curves("auc", "first")
            plot_curves("auc", "current")
            plot_curves("auprc", "first")
            plot_curves("auprc", "current")

            plot_predictions(X_test, y_test, d_figs, method_per_phase[0]['FIGSClassifier'], phases, phase=0)
            plt.savefig(os.path.join(results_dir, "phase_1_predictions.png"))
            plt.close()
            plot_predictions(X_test, y_test, d_figs, method_per_phase[1]["FIGSClassifier"], phases, phase=1)
            plt.savefig(os.path.join(results_dir, "phase_2_predictions.png"))
            plt.close()
            plot_predictions(X_test, y_test, d_figs, method_per_phase[2]["FIGSClassifier"], phases, phase=2)
            plt.savefig(os.path.join(results_dir, "phase_2_predictions.png"))
            plt.close()

    # save aucs to pickle
    with open(f"{results_dir}/results.pkl", "wb") as f:
        pickle.dump(results, f)


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
                ax.set_ylim(0.4, 1)

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
               y_test: pd.Series) -> dict:
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

    Returns:
        dict: AUC and AUPRC for each phase for each method

    """
    phases = get_phases(X_test.columns)
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


def main():
    run_sim(20, os.path.join("results", "dynamic_figs", "csi_pecarn"))
    plot_performance_results(os.path.join("results", "dynamic_figs", "csi_pecarn"))

if __name__ == '__main__':
    main()
