import logging
import nd2py as nd
import numpy as np
from typing import List, Dict

_logger = logging.getLogger(__name__)


def remove_repeat_items(
    items: List[nd.Symbol], X: Dict[str, np.array], edge_list, num_nodes,
    n_digits=7
) -> List[nd.Symbol]:
    """移除重复的项"""
    eval_args = dict(edge_list=edge_list, num_nodes=num_nodes)
    unique_items = []
    unique_values = []
    for item in items:
        val = item.eval(X, **eval_args)
        try:
            val = np.asarray(val, dtype=float)
            val = np.round(val, n_digits)
        except Exception as e:  # TODO: 如果之后没再看到这个 bug 就删了吧
            _logger.error(f"{e} (val={val}, item={item})")
        for val2 in unique_values:
            if np.array_equal(val, val2, equal_nan=True):
                break
        else:
            unique_items.append(item)
            unique_values.append(val)
    return unique_items
