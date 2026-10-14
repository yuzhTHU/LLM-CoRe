import pandas as pd


def df2dict(df: pd.DataFrame, keep_shape = False) -> dict:
    """ 将包含两层 column (MultiIndex) 的 DataFrame 转换成字典形式 """
    X = {}
    for var, sub_df in df.T.groupby(level=0):
        sub_df = sub_df.T
        if sub_df.nunique().max() == 1 and not keep_shape:
            X[var] = sub_df.iloc[[0]].values.copy()
        else:
            X[var] = sub_df.values.copy()
    return X
