import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


def memory_card_to_gb(series):

    def convert_single(value):
        if pd.isna(value):
            return 0
        value = str(value).strip().upper()

        if 'TB' in value:
            num = float(value.replace('TB', '').strip())
            return num * 1024
        elif 'GB' in value:
            num = float(value.replace('GB', '').strip())
            return num
        else:
            return 0

    return series.apply(convert_single)


def add_kfold_target_encoding(train_df, test_df, column, target='price', n_splits=5):
    """
    Adds proper out-of-fold target encoding for a categorical column.
    No leakage, works perfectly with your current pipeline.
    """
    train = train_df.copy()
    test = test_df.copy()

    # Global mean as fallback
    global_mean = train[target].mean()

    # Out-of-fold encoding for train
    oof_train = np.zeros(len(train))
    oof_test = np.zeros(len(test))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train):
        tr = train.iloc[tr_idx]
        val = train.iloc[val_idx]

        means = tr.groupby(column)[target].mean()
        oof_train[val_idx] = val[column].map(means)

        # Accumulate test predictions
        oof_test += test[column].map(means).fillna(global_mean).values / n_splits

    # Fill remaining NaN with global mean
    oof_train = np.where(pd.isna(oof_train), global_mean, oof_train)
    oof_test = np.where(pd.isna(oof_test), global_mean, oof_test)

    new_col = f'{column}_target_enc'
    train[new_col] = oof_train
    test[new_col] = oof_test

    return train, test