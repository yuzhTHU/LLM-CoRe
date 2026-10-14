import logging
import numpy as np
import nd2py as nd
import pandas as pd
from typing import List, Dict
from abc import ABC, abstractmethod
from ..core import Individual, Triplet
from .frozen_base import FrozenMixin

_logger = logging.getLogger(__name__)


class BaselineIndivLoader(FrozenMixin, ABC):
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
    def create(cls, assistant) -> 'BaselineIndivLoader':
        sub_class = cls.SUBCLASS_REGISTRY.get("default")
        return sub_class(assistant)

    @abstractmethod
    def load(self) -> pd.DataFrame:
        raise NotImplementedError


@BaselineIndivLoader.register('default')
class BaselineIndivLoader_v0(BaselineIndivLoader):
    def load(self):
        coeff_str="1 + sin(t) + cos(t) + sin(2 * t) + cos(2 * t)"
        f_sst_str="x + readout(ENSO * h) + ENSO * x**2 + ENSO * x * readout(ENSO * h) + IOD * x**2 + aggr(sour(x))"
        f_wwv_str="h + x + aggr(sour(x))"
        variables = {
            "t": nd.Variable("t", nettype="scalar"),
            "x": nd.Variable("x", nettype="node"),
            "h": nd.Variable("h", nettype="node"),
            "ENSO": nd.Variable("ENSO", nettype="node"),
            "IOD": nd.Variable("IOD", nettype="node"),
        }
        baseline_triplet = Triplet(
            coeff=nd.parse(coeff_str, variables=variables),
            f_sst=nd.parse(f_sst_str, variables=variables),
            f_wwv=nd.parse(f_wwv_str, variables=variables),
        )
        baseline_indiv = Individual(
            uuid='baseline',
            n_iter=-1,
            source='baseline',
            parents=[],
            triplet=baseline_triplet,
            program=self.assistant.triplet_to_program(baseline_triplet),
            dynamics=self.assistant.triplet_to_dynamics(baseline_triplet),
        )
        return baseline_indiv
