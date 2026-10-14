import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def evaluate_extremes(PREDICTION, mode=('x', 'ENSO'), threshold=0.5) -> pd.DataFrame:
    table = (
        PREDICTION
        .reset_index()
        .pipe(lambda df: df.assign(stop=df['time'] + df['lead']))
        .set_index(['time', 'lead', 'stop'])
        .pivot_table(index='lead', columns='stop', values=[mode])[mode]
    )
    table = pd.DataFrame(
        np.where(table.values > threshold, 1, np.where(table.values < -threshold, -1, np.where(np.isnan(table.values), np.nan, 0))),
        index=table.index, columns=table.columns,
    )
    table = table.loc[:, table.iloc[0].notna()].fillna(0)
    true = table.iloc[0]
    preds = table.iloc[0:]

    labels = [-1, 0, 1]  # 统一类别顺序
    results = []
    for idx, pred in preds.iterrows():
        # 计算 3x3 混淆矩阵
        cm = confusion_matrix(true.values, pred.values, labels=labels)

        # 对于 -1 类
        tp_ln = cm[0, 0]
        precision_ln = (tp_ln + 0) / (cm[:, 0].sum() + 0 + 1e-12)
        recall_ln    = (tp_ln + 0) / (cm[0, :].sum() + 0 + 1e-12)
        f1_ln = 2 * precision_ln * recall_ln / (precision_ln + recall_ln + 1e-12)

        # 对于 +1 类
        tp_en = cm[2, 2]
        precision_en = (tp_en + 0) / (cm[:, 2].sum() + 0 + 1e-12)
        recall_en    = (tp_en + 0) / (cm[2, :].sum() + 0 + 1e-12)
        f1_en = 2 * precision_en * recall_en / (precision_en + recall_en + 1e-12)

        # 误判概率
        prob_ln_to_en = cm[0, 2] / (cm[0, :].sum() + 1e-12) # P(pred=1 | true=-1) = cm[0,2] / sum(true=-1)
        prob_en_to_ln = cm[2, 0] / (cm[2, :].sum() + 1e-12) # P(pred=-1 | true=1) = cm[2,0] / sum(true=1)

        results.append({
            "precision_LaNina": precision_ln,
            "recall_LaNina": recall_ln,
            "f1_LaNina": f1_ln,
            "precision_ElNino": precision_en,
            "recall_ElNino": recall_en,
            "f1_ElNino": f1_en,
            "P(pred=ElNino|true=LaNina)": prob_ln_to_en,
            "P(pred=LaNina|true=ElNino)": prob_en_to_ln,
            "confusion_matrix": cm  # 保留原始 3x3 矩阵方便检查
        })

    return pd.DataFrame(results, index=preds.index)