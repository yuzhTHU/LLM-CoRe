import re
import logging
import nd2py as nd
from typing import List
from functools import reduce
from abc import ABC, abstractmethod
from ..utils import log_exception
from ..core import Triplet, Dynamics
from .frozen_base import FrozenMixin

_logger = logging.getLogger(__name__)


class Triplet2Reduction(FrozenMixin, ABC):
    SUBCLASS_REGISTRY = {}

    def __init__(self, assistant):
        self.assistant = assistant

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls.SUBCLASS_REGISTRY[name] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, assistant):
        sub_class = cls.SUBCLASS_REGISTRY.get("default")
        return sub_class(assistant)

    @abstractmethod
    def __call__(self, triplet):
        """ 将太复杂的 triplet 三元组进行简化, 不需要可逆 """
        raise NotImplementedError


@Triplet2Reduction.register('default')
class Triplet2Reduction_v0(Triplet2Reduction):
    def __call__(self, triplet):
        ## 移除自我嵌套的非线性函数
        simplify_args = dict(
            transform_constant_subtree=True,
            remove_useless_readout=True,
            remove_nested_sin=True,
            remove_nested_cos=True,
            remove_nested_tanh=True,
            remove_nested_sigmoid=True,
            remove_nested_sqrt=True,
            remove_nested_sqrtabs=True,
            remove_nested_exp=True,
            remove_nested_log=True,
            remove_nested_logabs=True,
        )
        coeff = triplet.coeff.simplify(**simplify_args)
        f_sst = triplet.f_sst.simplify(**simplify_args)
        f_wwv = triplet.f_wwv.simplify(**simplify_args)
        ## 重新合并同类项，调整变量类型
        coeff_items = self.merge_split_items(coeff)
        f_sst_items = self.merge_split_items(f_sst)
        f_wwv_items = self.merge_split_items(f_wwv)
        # coeff_items = self.remove_repeat_items(coeff_items, df)
        # f_sst_items = self.remove_repeat_items(f_sst_items, df)
        # f_wwv_items = self.remove_repeat_items(f_wwv_items, df)
        f_sst_items = self.merge_G_items(f_sst_items)
        f_wwv_items = self.merge_G_items(f_wwv_items)
        coeff = reduce(lambda a, b: a + b, coeff_items)
        f_sst = reduce(lambda a, b: a + b, f_sst_items)
        f_wwv = reduce(lambda a, b: a + b, f_wwv_items)
        ## coeff 中不应出现 x 和 h，f_sst 和 f_wwv 中不应出现 t
        for item in coeff.iter_preorder():
            if isinstance(item, nd.Variable) and item.name not in ['t']:
                item.name, item.nettype = 't', 'scalar'
        # 找一个可用的 t 以外的 scalar 变量
        scalar_var_list = [v for k, v in self.assistant.variables.items() if k != 't' and v.nettype == 'scalar']
        if len(scalar_var_list):
            scalar_var = scalar_var_list[0]
            # 用这个 scalar 变量替换 f_sst 和 f_wwv 中的 t 变量
            for item in f_sst.iter_preorder():
                if isinstance(item, nd.Variable) and item.name in ['t']:
                    item.name, item.nettype = scalar_var.name, scalar_var.nettype
            for item in f_wwv.iter_preorder():
                if isinstance(item, nd.Variable) and item.name in ['t']:
                    item.name, item.nettype = scalar_var.name, scalar_var.nettype
        else:
            # 用这个 scalar 变量替换 f_sst 和 f_wwv 中的 t 变量
            f_sst.parent = None
            for item in f_sst.iter_preorder():
                if isinstance(item, nd.Variable) and item.name in ['t']:
                    f_sst.replace(item, nd.Readout(self.assistant.variables['x'], nettype='scalar'))
            f_wwv.parent = None
            for item in f_wwv.iter_preorder():
                if isinstance(item, nd.Variable) and item.name in ['t']:
                    f_wwv.replace(item, nd.Readout(self.assistant.variables['x'], nettype='scalar'))
        return Triplet(coeff=coeff, f_sst=f_sst, f_wwv=f_wwv)

    @staticmethod
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

    @staticmethod
    def merge_G_items(items: List[nd.Symbol]) -> List[nd.Symbol]:
        op = None
        F_items = []
        G_items = []
        for item in items:
            if isinstance(item, nd.Aggr):
                op = nd.Aggr
                G_items.append(item.operands[0])
            elif isinstance(item, nd.Rgga):
                # G_items.append(item.operands[0])
                # op = ???
                F_items.append(item)
            elif isinstance(item, nd.Readout):
                if op is None: 
                    op = nd.Readout
                if item.operands[0].nettype in ['edge', 'scalar']:
                    G_items.append(item.operands[0])
                elif item.operands[0].nettype == 'node':
                    G_items.append(nd.sour(item.operands[0]))
                else:
                    raise ValueError(f"Unknown nettype in Readout: {item.operands[0].nettype}") 
            elif isinstance(item, (nd.Sour, nd.Targ)):
                raise ValueError("Sour and Targ are not allowed in f_sst")
            else:
                F_items.append(item)
        items = F_items
        if len(G_items):
            items.append(op(reduce(lambda a, b: a + b, G_items)))
        return items
