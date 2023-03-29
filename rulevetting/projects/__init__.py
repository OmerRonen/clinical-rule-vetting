import logging

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)

def _one_hot_encode_with_na(vec:pd.Series):
    if len(set(vec.values)) == 2:
        return vec
    # create one-hot encoding without "NA" category
    one_hot = pd.get_dummies(vec, dummy_na=False)

    # set all one-hot encoded values in rows with missing values to NaN
    one_hot = one_hot.mask(vec.isna(), np.nan)
    one_hot = one_hot.add_prefix(f"{vec.name}_")
    return one_hot

def one_hot_encode_df(df, numeric_cols):
    """Transforms categorical features in dataframe 
    Returns 
    -------
    one_hot_df: pd.DataFrame - categorical vars are one-hot encoded 
    """
    # grab categorical cols with >2 unique features
    # categorical_cols = [
    #     col for col in df.columns if not set(df[col].unique()).issubset({0.0, 1.0, 0, 1}) and col not in numeric_cols]
    # one_hot_df = pd.get_dummies(df.astype(str), columns=categorical_cols)

    one_hot_df = pd.concat([_one_hot_encode_with_na(df[col]) for col in df.columns if col not in numeric_cols], axis=1)
    
    return pd.concat([one_hot_df, df.loc[:, numeric_cols]], axis=1)
