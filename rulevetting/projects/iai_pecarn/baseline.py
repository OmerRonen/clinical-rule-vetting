import numpy as np
import pandas as pd

from rulevetting.templates.model import ModelTemplate


class Baseline(ModelTemplate):
    def __init__(self):
        # query for each rule + resulting predicted probability
        self.rules = [
            ('AbdTrauma_or_SeatBeltSign_yes == 1', 5.7),
            ('GCSScore < 14', 4.6),
            ('AbdTenderDegree_unknown == 0', 1.4),
            ('ThoracicTrauma_yes == 1', 0.6),
            ('AbdomenPain_yes == 1', 0.7),
            ('DecrBreathSound_yes == 1', 2.9),
            ('VomitWretch_yes == 1', 0.5),

            # final condition is just something that is always true
            ('GCSScore == GCSScore', 0.1),
        ]

    def _traverse_rule(self, df_features: pd.DataFrame):
        str_print = f''
        predicted_probabilities = pd.Series(index=df_features.index, dtype=float)
        df = df_features.copy()
        o = 'outcome'
        str_print += f'{df[o].sum()} / {df.shape[0]} (positive class / total)\n\t\u2193 \n'
        for j, rule in enumerate(self.rules):
            query, prob = rule
            df_rhs = df.query(query)
            idxs_satisfying_rule = df_rhs.index
            predicted_probabilities.loc[idxs_satisfying_rule] = prob
            df.drop(index=idxs_satisfying_rule, inplace=True)
            computed_prob = 100 * df_rhs[o].sum() / df_rhs.shape[0]
            query_print = query.replace(' == 1', '')
            if j < len(self.rules) - 1:
                str_print += f'\033[96mIf {query_print:<35}\033[00m \u2192 {df_rhs[o].sum():>3} / {df_rhs.shape[0]:>4} ({computed_prob:0.1f}%)\n\t\u2193 \n   {df[o].sum():>3} / {df.shape[0]:>5}\t \n'
        # print(predicted_probabilities)
        predicted_probabilities = predicted_probabilities.values
        self.str_print = str_print
        return predicted_probabilities

    def predict(self, df_features: pd.DataFrame):
        predicted_probabilities = self._traverse_rule(df_features)
        return (predicted_probabilities > 0.11).astype(int)

    def predict_proba(self, df_features: pd.DataFrame):
        predicted_probabilities = self._traverse_rule(df_features) / 100
        return np.vstack((1 - predicted_probabilities, predicted_probabilities)).transpose()

    def print_model(self, df_features):
        self._traverse_rule(df_features)
        return self.str_print

if __name__ == '__main__':
    from rulevetting.projects.iai_pecarn.dataset import Dataset
    # df_train, df_tune, df_test = Dataset().get_data(load_csvs=True) # if there are processed data in /data/csi_pecarn/processed
    df_train, df_tune, df_test = Dataset().get_data()
    df_full = pd.concat((df_train, df_tune, df_test))
    model = Baseline()
    preds_proba = model.predict_proba(df_full)
    print(model.print_model(df_full))
    print(preds_proba)