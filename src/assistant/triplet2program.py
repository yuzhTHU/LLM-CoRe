import re
import logging
import nd2py as nd
from abc import ABC, abstractmethod
from ..utils import log_exception
from .frozen_base import FrozenMixin

_logger = logging.getLogger(__name__)


class Triplet2Program(FrozenMixin, ABC):
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
        """ 将 triplet 三元组转换成对应的 program 代码, 除常数外需要完全保证可逆 """
        sub_class = cls.SUBCLASS_REGISTRY.get("default")
        return sub_class(assistant)

    @abstractmethod
    def __call__(self, triplet):
        raise NotImplementedError


@Triplet2Program.register('default')
class Triplet2Program_v1(Triplet2Program):
    def __call__(self, triplet):
        coeff = triplet.coeff.to_str(number_format='.2f')
        f_sst = triplet.f_sst.to_str(number_format='.2f')
        f_wwv = triplet.f_wwv.to_str(number_format='.2f')
        ## Replace climate modes with mode['climate mode']
        for mode in self.assistant.modes:
            coeff = re.sub(rf"\b{mode}\b", lambda x: f"mode['{x.group(0)}']", coeff)
            f_sst = re.sub(rf"\b{mode}\b", lambda x: f"mode['{x.group(0)}']", f_sst)
            f_wwv = re.sub(rf"\b{mode}\b", lambda x: f"mode['{x.group(0)}']", f_wwv)
        ## Create the program string
        program = (
            f"{self.assistant.signature}\n"
            f"    coeff = {coeff}\n"
            f"    dx_dt = {f_sst}\n"
            f"    dh_dt = {f_wwv}\n"
            f"    return coeff, dx_dt, dh_dt"
        )
        return program


@Triplet2Program.register('use_C')
class Triplet2Program_v2(Triplet2Program):
    def __call__(self, triplet):
        printer = nd.StringPrinter()
        printer.visit_Number = self.visit_Number_as_C
        coeff = printer(triplet.coeff, skeleton=True)
        f_sst = printer(triplet.f_sst, skeleton=True)
        f_wwv = printer(triplet.f_wwv, skeleton=True)
        cnt = range(100).__iter__()
        coeff = re.sub(r"\bC\b", lambda x: f"nl_params[{next(cnt)}]", coeff)
        f_sst = re.sub(r"\bC\b", lambda x: f"nl_params[{next(cnt)}]", f_sst)
        f_wwv = re.sub(r"\bC\b", lambda x: f"nl_params[{next(cnt)}]", f_wwv)
        ## Replace climate modes with mode['climate mode']
        for mode in self.assistant.modes:
            coeff = re.sub(rf"\b{mode}\b", lambda x: f"mode['{x.group(0)}']", coeff)
            f_sst = re.sub(rf"\b{mode}\b", lambda x: f"mode['{x.group(0)}']", f_sst)
            f_wwv = re.sub(rf"\b{mode}\b", lambda x: f"mode['{x.group(0)}']", f_wwv)
        ## Create the program string
        program = (
            f"{self.assistant.signature}\n"
            f"    coeff = {coeff}\n"
            f"    dx_dt = {f_sst}\n"
            f"    dh_dt = {f_wwv}\n"
            f"    return coeff, dx_dt, dh_dt"
        )
        return program

    @staticmethod
    def visit_Number_as_C(node, *args, **kwargs):
        yield from []
        try:
            if int(node.value) == node.value or node.value == 1:
                return f"{int(node.value)}"
            return "C"
        except Exception as e:
            _logger.error(f"visit_Number(node={node.value}) error since: {log_exception(e)}")
            return "C"
