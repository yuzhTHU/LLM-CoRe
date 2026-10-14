import re
import logging
import numbers
import nd2py as nd
import numpy as np
from nd2py.core.basic.get_copy import GetCopy
from nd2py.core.calc.numpy_calc import unpack_operands
from functools import reduce
from abc import ABC, abstractmethod
from ..utils import log_exception
from ..core import Triplet, Dynamics
from .frozen_base import FrozenMixin

_logger = logging.getLogger(__name__)


# nd2py 2.4.0 treats every one-element ndarray as a scalar in graph
# aggregation.  NumPy 2.x no longer permits converting a non-0d array with
# float(), and batched (..., 1) values must follow the broadcast branch below.
@unpack_operands()
def _numpy_calc_visit_aggr(self, node, x, *args, **kwargs):
    edge_list = kwargs.get("edge_list", ([], []))
    num_nodes = kwargs.get("num_nodes")
    if isinstance(x, numbers.Number) or x.ndim == 0:
        y = np.zeros((num_nodes,))
        np.add.at(y, edge_list[1], float(x))
        return y
    if node.operands[0].nettype == "scalar" or x.shape[-1] == 1:
        if x.shape[-1] != 1:
            x = x[..., np.newaxis]
        y = np.zeros((num_nodes,))
        np.add.at(y, edge_list[1], 1)
        return y * x
    y = np.zeros((*x.shape[:-1], num_nodes))
    for k, j in enumerate(edge_list[1]):
        y[..., j] += x[..., k]
    return y


@unpack_operands()
def _numpy_calc_visit_rgga(self, node, x, *args, **kwargs):
    edge_list = kwargs.get("edge_list", ([], []))
    num_nodes = kwargs.get("num_nodes")
    if isinstance(x, numbers.Number) or x.ndim == 0:
        y = np.zeros((num_nodes,))
        np.add.at(y, edge_list[1], x)
        return y
    if node.operands[0].nettype == "scalar" or x.shape[-1] == 1:
        if x.shape[-1] != 1:
            x = x[..., np.newaxis]
        y = np.zeros((num_nodes,))
        np.add.at(y, edge_list[0], 1)
        return y * x
    y = np.zeros((*x.shape[:-1], num_nodes))
    for k, i in enumerate(edge_list[0]):
        y[..., i] += x[..., k]
    return y


nd.NumpyCalc.visit_Aggr = _numpy_calc_visit_aggr
nd.NumpyCalc.visit_Rgga = _numpy_calc_visit_rgga


class Triplet2Dynamics(FrozenMixin, ABC):
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
        sub_class = cls.SUBCLASS_REGISTRY.get("set_number_name")
        return sub_class(assistant)

    @abstractmethod
    def __call__(self, triplet):
        """ 将 triplet 三元组转换成 Dynamics 对象, 需要完全保证可逆 """
        raise NotImplementedError


@Triplet2Dynamics.register('default')
class Triplet2Dynamics_v0(Triplet2Dynamics):
    def __call__(self, triplet):
        # Set coeff_sst and coeff_wwv
        sst_items = triplet.f_sst.split_by_add()
        wwv_items = triplet.f_wwv.split_by_add()
        coeff_items = triplet.coeff.split_by_add()
        sst_items_with_coeff = []
        for sst_item in sst_items:
            for coeff_item in coeff_items:
                if isinstance(sst_item, (nd.Aggr, nd.Rgga, nd.Readout)):
                    sst_items_with_coeff.append(type(sst_item)(coeff_item * sst_item.operands[0]))
                elif isinstance(sst_item, (nd.Sour, nd.Targ)):
                    raise ValueError("Sour and Targ are not allowed in f_sst")
                else:
                    sst_items_with_coeff.append(sst_item * coeff_item)
        wwv_items_with_coeff = []
        for wwv_item in wwv_items:
            for coeff_item in coeff_items:
                if isinstance(wwv_item, (nd.Aggr, nd.Rgga, nd.Readout)):
                    wwv_items_with_coeff.append(type(wwv_item)(coeff_item * wwv_item.operands[0]))
                elif isinstance(wwv_item, (nd.Sour, nd.Targ)):
                    raise ValueError("Sour and Targ are not allowed in f_wwv")
                else:
                    wwv_items_with_coeff.append(wwv_item * coeff_item)
        ddt_sst = reduce(lambda a, b: a + b, sst_items_with_coeff)
        ddt_wwv = reduce(lambda a, b: a + b, wwv_items_with_coeff)
        return Dynamics(ddt_sst=ddt_sst, ddt_wwv=ddt_wwv)


class NamedNumber(nd.Number):
    def __init__(self, name, value, nettype='scalar', fitable=False):
        super().__init__(value, nettype, fitable)
        self.name = name
    
    def GetCopy_visit_NamedNumber(self, node, *args, **kwargs):
        yield from []
        return node.__class__(
            node.name, node.value, nettype=node.nettype, fitable=node.fitable
        )
    GetCopy.visit_NamedNumber = GetCopy_visit_NamedNumber

    # def StringPriner_visit_NamedNumber(self, node, *args, **kwargs):
    #     string = nd.StringPrinter.visit_Number(self, node, *args, **kwargs)
    #     return string + f'{{{node.name}}}'
    
    nd.StringPrinter.visit_NamedNumber = nd.StringPrinter.visit_Number

    nd.NumpyCalc.visit_NamedNumber = nd.NumpyCalc.visit_Number


def replace_number_with_named_number(eq: nd.Symbol, idx):
    eq = eq.copy()
    source = str(eq)
    replacements = []
    for number in eq.iter_preorder():
        if isinstance(number, nd.Number):
            replacements.append((eq.path_to(number), number.value, number.nettype, number.fitable))
    # Symbol.replace() locates a parent's child with list.index(), which is
    # ambiguous when an expression contains equal-valued Number siblings.
    # Replace by identity-derived paths so every duplicate coefficient is
    # named exactly once.
    for path, value, nettype, fitable in replacements:
        named_number = NamedNumber(f'{idx} :: {source} :: {path}', value, nettype, fitable)
        if not path:
            eq = named_number
            continue
        parent = eq.get_path(path[:-1])
        old = parent.operands[path[-1]]
        parent.operands[path[-1]] = named_number
        named_number.parent = parent
        old.parent = None
    return eq


@Triplet2Dynamics.register('set_number_name')
class Triplet2Dynamics_v1(Triplet2Dynamics):
    def __call__(self, triplet):
        # Set coeff_sst and coeff_wwv
        sst_items = triplet.f_sst.split_by_add()
        wwv_items = triplet.f_wwv.split_by_add()
        coeff_items = triplet.coeff.split_by_add()
        sst_items_with_coeff = []
        for sst_idx, sst_item in enumerate(sst_items):
            for coeff_idx, coeff_item in enumerate(coeff_items):
                if isinstance(sst_item, (nd.Aggr, nd.Rgga, nd.Readout)):
                    op = type(sst_item)
                    _sst_item = replace_number_with_named_number(sst_item.operands[0], sst_idx)
                    _coeff_item = replace_number_with_named_number(coeff_item, coeff_idx)
                    sst_items_with_coeff.append(op(_sst_item * _coeff_item))
                elif isinstance(sst_item, (nd.Sour, nd.Targ)):
                    raise ValueError("Sour and Targ are not allowed in f_sst")
                else:
                    _sst_item = replace_number_with_named_number(sst_item, sst_idx)
                    _coeff_item = replace_number_with_named_number(coeff_item, coeff_idx)
                    sst_items_with_coeff.append(_sst_item * _coeff_item)
        wwv_items_with_coeff = []
        for wwv_idx, wwv_item in enumerate(wwv_items):
            for coeff_idx, coeff_item in enumerate(coeff_items):
                if isinstance(wwv_item, (nd.Aggr, nd.Rgga, nd.Readout)):
                    op = type(wwv_item)
                    _wwv_item = replace_number_with_named_number(wwv_item.operands[0], wwv_idx)
                    _coeff_item = replace_number_with_named_number(coeff_item, coeff_idx)
                    wwv_items_with_coeff.append(op(_wwv_item * _coeff_item))
                elif isinstance(wwv_item, (nd.Sour, nd.Targ)):
                    raise ValueError("Sour and Targ are not allowed in f_wwv")
                else:
                    _wwv_item = replace_number_with_named_number(wwv_item, wwv_idx)
                    _coeff_item = replace_number_with_named_number(coeff_item, coeff_idx)
                    wwv_items_with_coeff.append(_wwv_item * _coeff_item)
        ddt_sst = reduce(lambda a, b: a + b, sst_items_with_coeff)
        ddt_wwv = reduce(lambda a, b: a + b, wwv_items_with_coeff)
        return Dynamics(ddt_sst=ddt_sst, ddt_wwv=ddt_wwv)
