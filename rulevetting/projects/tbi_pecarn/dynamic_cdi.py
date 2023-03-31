import copy
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from imodels.tree.dfigs import D_FIGSClassifier
from imodels.tree.figs import FIGSClassifier
from imodels import RuleFitClassifier, BayesianRuleListClassifier, GreedyRuleListClassifier
from mlxtend.classifier import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from rulevetting.projects.tbi_pecarn.dataset import Dataset as tbiDataset

METHODS = [FIGSClassifier, RandomForestClassifier, DecisionTreeClassifier]


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


def get_tbi_data():
    data_train, data_tune, data_test = tbiDataset().get_data(seed=11)
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
    return X.loc[idx_first, :], y[idx_first]


def get_phase_indices(data, phase_features):
    idx = data.iloc[:, phase_features].notna().all(axis=1)
    return idx


def _get_phase(i, phases):
    for phase in phases:
        if i in phases[phase]:
            return phase + 1

    return "NA"


def fit_methods(X_train, y_train, phases, methods):
    method_per_phase = {phase: {} for phase in phases}
    for phase, vars in phases.items():
        idx = get_phase_indices(X_train, vars)
        X_train_phase = X_train.iloc[:, vars]

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
    ax.scatter(preds_dfigs[idx_true], preds_method[idx_true], c='r', s=10 * y_test[idx_true] + 1, label="Tbi")
    # now predictions on false cases
    idx_false = y_test == 0
    ax.scatter(preds_dfigs[idx_false], preds_method[idx_false], c='b', s=10 * y_test[idx_false] + 1, label="No Tbi")
    ax.set_xlabel(d_figs.__class__.__name__)
    method_name = method.__class__.__name__
    ax.set_ylabel(method_name)
    ax.set_title(f"Phase {phase}")
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


def run_sim(n_seeds, max_rules=12):
    X, y = get_tbi_data()
    phases = get_phases(X.columns)
    for phase in phases:
        print(f"phase {phase} shape: {get_phase_indices(X, phases[phase]).sum()}")

    features_names = [f"{f}_phase_{_get_phase(i, phases)}" for i, f in enumerate(X.columns)]
    methods_dict = {m.__name__:[] for m in METHODS+ [D_FIGSClassifier]}
    results_template = {phase:{"auc":copy.deepcopy(methods_dict), "auprc":copy.deepcopy(methods_dict)} for phase in phases}
    results = {"current": copy.deepcopy(results_template), "first": copy.deepcopy(results_template)}
    for seed in range(n_seeds):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)
        print(np.sum(y_test))


        d_figs = D_FIGSClassifier(phases=phases, max_rules=max_rules)
        d_figs.feature_names_ = features_names
        d_figs.fit(X_train.values, y_train.values)

        method_per_phase = fit_methods(X_train, y_train, phases, METHODS)
        scores_seed_first = get_scores("first", phases, method_per_phase, d_figs, X_test, y_test)
        scores_seed_current = get_scores("current", phases, method_per_phase, d_figs, X_test, y_test)
        #
        for phase in phases:

            for m in scores_seed_current["auroc"][phase]:
                results["current"][phase]["auc"][m].append(scores_seed_current["auroc"][phase][m])
                results["current"][phase]["auprc"][m].append(scores_seed_current["auprc"][phase][m])

                results["first"][phase]["auc"][m].append(scores_seed_first["auroc"][phase][m])
                results["first"][phase]["auprc"][m].append(scores_seed_first["auprc"][phase][m])

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
                preds = {m: clf.predict_proba(X_test_phase.loc[phase_idx, :].values)[:, 1] for m, clf in phase_methods.items()}
                if phase > 0:
                    preds[d_figs.__class__.__name__] = d_figs.predict_proba(X_test.loc[phase_idx, :].values)[:, 1]
                plt_func(preds, y_test[phase_idx], ax=axs[phase])
                axs[phase].set_title(f"Phase {phase} (# of patients: {phase_idx.sum()})")
                loc = "lower right" if curve_type == "auc" else "upper right"
                axs[phase].legend(loc=loc)
            plt.suptitle(title)
            plt.savefig(f"results/dynamic_figs/{fig_name}")

        if seed == 1:
            plot_curves("auc", "first")
            plot_curves("auc", "current")
            plot_curves("auprc", "first")
            plot_curves("auprc", "current")

            plot_predictions(X_test, y_test, d_figs, method_per_phase[0]['FIGSClassifier'], phases, phase=0)
            plt.savefig("results/dynamic_figs/phase_1_predictions.png")
            plt.close()
            plot_predictions(X_test, y_test, d_figs, method_per_phase[1]["FIGSClassifier"], phases, phase=1)
            plt.savefig("results/dynamic_figs/phase_2_predictions.png")
            plt.close()
            plot_predictions(X_test, y_test, d_figs, method_per_phase[2]["FIGSClassifier"], phases, phase=2)
            plt.savefig("results/dynamic_figs/phase_3_predictions.png")
            plt.close()


            # fig, axs = plt.subplots(1, 3, figsize=(15, 15))
            # for phase, phase_vars in phases.items():
            #     phase_methods = method_per_phase[phase]
            #     phase_idx = get_phase_indices(X_test, phase_vars)
            #     if phase + 1 in phases:
            #         phase_idx = np.logical_and(phase_idx, np.invert(get_phase_indices(X_test, phases[phase + 1])))
            #     X_test_phase = X_test.iloc[:, phase_vars]
            #     preds = {m: clf.predict_proba(X_test_phase.loc[phase_idx, :].values)[:, 1] for m, clf in phase_methods.items()}
            #     if phase > 0:
            #         preds[d_figs.__class__.__name__] = d_figs.predict_proba(X_test.loc[phase_idx, :].values)[:, 1]
            #     plot_prs(preds, y_test[phase_idx], ax=axs[phase])
            #     axs[phase].set_title(f"Phase {phase}")
            #     axs[phase].legend(loc="lower right")
            # plt.suptitle("PRC curves for different methods")
            # plt.show()

    # save aucs to pickle
    with open("results/dynamic_figs/results.pkl", "wb") as f:
        pickle.dump(results, f)

def plot_performance_results():
    # load results from pickle
    with open("results/dynamic_figs/results.pkl", "rb") as f:
        results = pickle.load(f)
    # make bar plots for auc and aucpr for each phase for current key in results
    def save_bar_plot(score_type):
        for key, scores in results.items():
            fig, axs = plt.subplots(1, 3, figsize=(15, 15))
            for phase, phase_scores in scores.items():
                # fig, ax = plt.subplots(1, 1, figsize=(15, 15))
                for method, scores in phase_scores[score_type].items():
                    std_mean = 2 * np.std(scores) / np.sqrt(len(scores))
                    axs[phase].bar(method, np.mean(scores), yerr=std_mean, label=method)

                axs[phase].set_title(f"phase {phase}")
                axs[phase].legend()
                # remove xticks
                axs[phase].set_xticks([])
            plt.savefig(f"results/dynamic_figs/{key}_{score_type}_bar.png")
            plt.close()

    save_bar_plot("auc")
    save_bar_plot("auprc")
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





def get_scores(compare, phases, method_per_phase, d_figs, X_test, y_test):
    aucs_phase = {}
    aucprs_phase = {}
    for phase in phases:
        methods_phase = phase if compare != "first" else 0
        phase_idx = get_phase_indices(X_test, phases[phase])
        if phase + 1 in phases and methods_phase != "first":
            phase_idx = np.logical_and(phase_idx, np.invert(get_phase_indices(X_test, phases[phase + 1])))
        X_test_phase = X_test.iloc[:, phases[methods_phase]]
        preds = {name: method.predict_proba(X_test_phase.loc[phase_idx, :].values)[:, 1] for name, method in
                 method_per_phase[methods_phase].items()}
        if phase > 0:
            d_figs_preds = d_figs.predict_proba(X_test.loc[phase_idx, :].values)[:, 1]
            preds["D_FIGSClassifier"] = d_figs_preds

        aucs_phase[phase] = get_aucs(preds, y_test[phase_idx])
        aucprs_phase[phase] = get_aps(preds, y_test[phase_idx])

    return {"auroc": aucs_phase, "auprc": aucprs_phase}


def make_dfigs_tree_plots():
    X, y = get_tbi_data()
    # split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

    phases = get_phases(X.columns)
    features_names = [f"{f}_phase_{_get_phase(i, phases)}" for i, f in enumerate(X.columns)]

    d_figs_true = D_FIGSClassifier(phases=phases, max_rules=2)
    d_figs_true.fit(X_train.values, y_train.values)
    d_figs_true.feature_names_ = features_names

    # calculate auroc for d_figs_true
    auroc = roc_auc_score(y_test, d_figs_true.predict_proba(X_test.values)[:, 1])
    print(f"auROC for D_FIGSClassifier: {auroc}")

    d_figs_true.plot(filename="results/dynamic_figs/d_figs_true.png", label="none")

    d_figs_perturbed = D_FIGSClassifier(phases=phases, max_rules=2)
    # fit on permuted labels
    d_figs_perturbed.fit(X_train.values, np.random.permutation(y))
    d_figs_perturbed.feature_names_ = features_names
    d_figs_perturbed.plot(filename="results/dynamic_figs/d_figs_permuted_labels.png", label="none")
    # calculate auroc for d_figs_true
    auroc = roc_auc_score(y_test, d_figs_perturbed.predict_proba(X_test.values)[:, 1])
    print(f"auROC for D_FIGSClassifier: {auroc}")

    # I want to swap the locations of 'AMS_1.0_phase_3' and 'Drugs_1.0_phase_3' in feature names
    i_1 = features_names.index('AMS_1.0_phase_3')
    i_2 = features_names.index('Drugs_1.0_phase_3')
    features_names[i_1] = 'Drugs_1.0_phase_3'
    features_names[i_2] = 'AMS_1.0_phase_3'
    # now do the same with 'HA_verb_91.0_phase_3' and 'SFxPalp_Yes_phase_3'
    i_1 = features_names.index('HA_verb_91.0_phase_3')
    i_2 = features_names.index('SFxPalp_Yes_phase_3')
    features_names[i_1] = 'SFxPalp_Yes_phase_3'
    features_names[i_2] = 'HA_verb_91.0_phase_3'


    d_figs_true.feature_names_ = features_names
    d_figs_true.plot(filename="results/dynamic_figs/d_figs_permuted_vars.png", label="none")





def main():
    # run_sim(20)
    # plot_performance_results()
    make_dfigs_tree_plots()

    # # # (10)


if __name__ == '__main__':
    main()
