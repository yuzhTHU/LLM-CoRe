import pandas as pd


def split_ab(df: pd.DataFrame, dt: float = None):
    if dt is None:
        dt = df[('t', 'value')].diff(-1).mul(-1).min()
        mask = (df[('t', 'value')].diff(-1).mul(-1) - dt).abs() < 1e-6
    else:
        mask = None
    A = df.iloc[:-1]
    b_sst = df['x'].diff(-1).mul(-1).iloc[:-1] / dt
    b_wwv = df['h'].diff(-1).mul(-1).iloc[:-1] / dt
    if mask is not None:
        A = A[mask.iloc[:-1].values]
        b_sst = b_sst[mask.iloc[:-1].values]
        b_wwv = b_wwv[mask.iloc[:-1].values]
    return A, b_sst, b_wwv
