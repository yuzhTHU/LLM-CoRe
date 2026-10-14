import logging
import nd2py as nd
from functools import reduce
from typing import List, Dict

_logger = logging.getLogger(__name__)


def merge_split_items(func: nd.Symbol) -> List[nd.Symbol]:
    """将 func 拆成多项式，把 aggr 提到乘法外面"""
    # 将 func 拆成多项式
    items = func.split_by_add(
        split_by_sub=True,
        expand_mul=True,
        expand_div=True,
        expand_aggr=True,
        expand_sour=True,
        expand_targ=True,
        expand_readout=True,
        remove_coefficients=True,
        merge_bias=True,
    )
    # 将 aggr 提到乘法外面：aggr(a) * b -> aggr(a * b), readout(a) * b -> readout(a * b)
    for idx, item in enumerate(items):
        factors = item.split_by_mul()
        if len(factors) <= 1:
            # 只有一个因子，不需要修改
            continue
        elif any(is_aggr := [isinstance(f, (nd.Aggr, nd.Rgga)) for f in factors]):
            # 将第一个 aggr 因子的 aggr 提到外面，其它因子转换为 edge
            aggr_idx = is_aggr.index(True)
            symbol = type(factors[aggr_idx])  # Aggr or Rgga
            for i, factor in enumerate(factors):
                if i == aggr_idx:
                    factors[i] = factor.operands[0]
                elif factor.nettype == "scalar":
                    pass
                else:
                    factors[i] = nd.targ(factor)
            items[idx] = symbol(reduce(lambda a, b: a * b, factors))
        elif item.nettype == "scalar" and any(
            is_readout := [isinstance(f, nd.Readout) for f in factors]
        ):
            # 如果其他因子都是 scalar, 将第一个 readout 因子的 readout 提到外面
            readout_idx = is_readout.index(True)
            factors[readout_idx] = factors[readout_idx].operands[0]
            items[idx] = nd.readout(reduce(lambda a, b: a * b, factors))
    return items

