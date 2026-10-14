import re
import logging
import nd2py as nd
from functools import reduce
from abc import ABC, abstractmethod
from ..utils import log_exception
from ..core import Triplet, Dynamics
from .frozen_base import FrozenMixin

_logger = logging.getLogger(__name__)


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
    nd.GetCopy.visit_NamedNumber = GetCopy_visit_NamedNumber

    # def StringPriner_visit_NamedNumber(self, node, *args, **kwargs):
    #     string = nd.StringPrinter.visit_Number(self, node, *args, **kwargs)
    #     return string + f'{{{node.name}}}'
    
    nd.StringPrinter.visit_NamedNumber = nd.StringPrinter.visit_Number

    nd.NumpyCalc.visit_NamedNumber = nd.NumpyCalc.visit_Number


def replace_number_with_named_number(eq: nd.Symbol, idx):
    eq = eq.copy()
    replace_dict = []
    for number in eq.iter_preorder():
        if isinstance(number, nd.Number):
            named_number = NamedNumber(f'{idx} :: {eq} :: {eq.path_to(number)}', number.value, number.nettype, number.fitable)
            replace_dict.append((number, named_number))
    for number, named_number in replace_dict:
        eq = eq.replace(number, named_number)
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
