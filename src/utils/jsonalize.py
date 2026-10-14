import numpy as np
import pandas as pd


def jsonalize(data):
    if isinstance(data, dict):
        return {
            ("/".join(map(str, k)) if isinstance(k, tuple) else str(jsonalize(k))): jsonalize(v)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [jsonalize(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(jsonalize(v) for v in data)
    elif isinstance(data, (np.integer,)):
        return int(data)
    elif isinstance(data, (np.floating,)):
        return float(data)
    elif isinstance(data, (np.ndarray,)):
        return jsonalize(data.tolist())
    elif isinstance(data, pd.Series):
        return jsonalize(data.to_dict())
    elif isinstance(data, pd.DataFrame):
        return jsonalize(data.to_dict(orient='records'))
    else:
        return data
