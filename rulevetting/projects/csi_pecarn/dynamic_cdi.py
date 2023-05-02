import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree


def get_phases(columns):
    df_label = pd.read_csv('data/csi_pecarn/csi_vars_with_labels.csv')
    # df_label = df_label.rename(
    #     columns={'Time (Aaron) 1= Prehospital, 2=primary survey, 3= first 1 hour, 4= > 1hour': 'timestamp'})
    time_distribution = df_label.groupby('Phase')['feature'].agg(list)

    phases = {0: [], 1: [], 2: []}  # 0: prehospital, 1: hospital

    phase_1_features = time_distribution[1]
    phase_2_features = time_distribution[1] + time_distribution[2]
    phase_3_features = time_distribution[1] + time_distribution[2] + time_distribution[3] #+ time_distribution[4]
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

    return phases


def single_split_example():
    x = np.arange(0, 12, 1)
    y = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1])
    y = 1-y
    # fit decision tree with a single split
    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(x.reshape(-1, 1), y)
    # plot the tree and the data in two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plot_tree(clf, ax=ax1, feature_names=["x"])
    ax1.set_title("Decision tree")
    ax2.scatter(x, y)
    ax2.set_title("Data")
    ax2.set_xlabel("Measurement")
    ax2.set_ylabel("Outcome")
    # make the y ticks be 1 = sick and 0 = healthy
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["healthy", "sick"])
    # add vertical red lines at the split point and label it "cart split"
    ax2.axvline(x=clf.tree_.threshold[0], color="red", linestyle="--", label="cart split")
    # add vertical green line at x=10 and label it possible alternative split
    ax2.axvline(x=8.5, color="green", linestyle="--", label="possible alternative split")
    ax2.legend()
    # save figure to the results folder under dynamic_figs
    fig.savefig("results/dynamic_figs/single_split_example.png", dpi=300)
    plt.close(fig)
