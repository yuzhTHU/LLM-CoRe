import gzip
import pickle
import logging
import nd2py as nd
import numpy as np
from typing import List
from abc import ABC, abstractmethod
from ..core import Individual, Triplet, Dynamics
from ..utils import my_hash, my_uuid
from .frozen_base import FrozenMixin

_logger = logging.getLogger(__name__)


class SeedIndividualsLoader(FrozenMixin, ABC):
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
    def create(cls, assistant) -> 'SeedIndividualsLoader':
        sub_class = cls.SUBCLASS_REGISTRY.get("default")
        return sub_class(assistant)

    @abstractmethod
    def load(self) -> List[Individual]:
        raise NotImplementedError


@SeedIndividualsLoader.register('default')
class SeedIndividualsLoader_v0(SeedIndividualsLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        variables = {
            "t": nd.Variable("t", nettype="scalar"),
            "x": nd.Variable("x", nettype="node"),
            "h": nd.Variable("h", nettype="node"),
            "ENSO": nd.Variable("ENSO", nettype="node"),
            "IOD": nd.Variable("IOD", nettype="node"),
        }
        self.triplet_dict = dict(
            linear=Triplet(
                coeff=nd.parse("1 + sin(t) + cos(t)", variables=variables),
                f_wwv=nd.parse("x + h + aggr(sour(x) + sour(h))", variables=variables),
                f_sst=nd.parse("x + h + aggr(sour(x) + sour(h))", variables=variables),
            ),
            XRO=Triplet(
                coeff=nd.parse("1 + sin(t) + cos(t) + sin(2 * t) + cos(2 * t)", variables=variables),
                f_sst=nd.parse("x + readout(ENSO * h) + ENSO * x**2 + ENSO * x * readout(ENSO * h) + IOD * x**2 + aggr(sour(x))", variables=variables),
                f_wwv=nd.parse("h + x + aggr(sour(x))", variables=variables),
            ),
            co2_30_14=Triplet(
                coeff=nd.parse("1+sin(t)+cos(t)+sin(0.2*t)", variables=variables),
                f_sst=nd.parse("1+x+readout(ENSO*h)+x**2+tanh(u_co2)+aggr(sour(x))", variables=variables),
                f_wwv=nd.parse("x+aggr(sour(x))", variables=variables),
            ),
            temp_25_30=Triplet(
                coeff=nd.parse("1 + sin(t) + cos(t) + cos(1.25e-01 * t) + sin(1.00e-01 * t) ** 4", variables=variables),
                f_sst=nd.parse("x + h + u_temp + aggr(sour(x) + sour(h) + sour(h))", variables=variables),
                f_wwv=nd.parse("h + x + u_temp * u_temp + aggr(sour(x))", variables=variables),
            ),
            temp_30_25=Triplet(
                coeff=nd.parse("1 + sigmoid(sin(t)) + cos(-8.00e-02 * t) + cos(2.50e-01 * t) + exp(-8.00e-02 * t)", variables=variables),
                f_sst=nd.parse("x + tanh(u_temp) + tanh(u_temp ** 2) + tanh(aggr(sour(x))) + aggr(sour(x))", variables=variables),
                f_wwv=nd.parse("h + u_co2 + sigmoid(u_co2) + aggr(sour(h))", variables=variables),
            ),
            aot_co2_15_30=Triplet(
                coeff=nd.parse("t + sin(t) + cos(t) + tanh(t) + sin(5.00e-01 * t)", variables=variables),
                f_sst=nd.parse("x + ENSO * tanh(7.00e-01 * h) + exp(u_co2**2) + aggr(sour(x) + sour(h))", variables=variables),
                f_wwv=nd.parse("1 + h + aggr(sigmoid(2.50e-01 * u_aot**2) * sour(h) + sour(x) * sqrtabs(sour(aggr(sour(x)))))", variables=variables),
            ),
            no_forcing_25_30=Triplet(
                coeff=nd.parse("1 + cos(t) + cos(1.25e-01 * t) + sin(1.00e-01 * t) ** 2", variables=variables),
                f_sst=nd.parse("x + h + aggr(sour(x) + sour(h))", variables=variables),
                f_wwv=nd.parse("x + h + aggr(sour(x))", variables=variables),
            ),
            co2_29_29=Triplet(
                coeff=nd.parse("1 + sin(t) + cos(t) + cos(1.25e-01 * t) + sin(1.00e-01 * t) ** 4", variables=variables),
                f_sst=nd.parse("x + h + aggr(sour(h))", variables=variables),
                f_wwv=nd.parse("h + x + u_co2 ** 5 + aggr(sour(x))", variables=variables),
            ),
            co2_30_30=Triplet(
                coeff=nd.parse("1 + cos(t) + cos(1.25e-01 * t) + sin(1.00e-01 * t) ** 4", variables=variables),
                f_sst=nd.parse("x + h + aggr(sour(h))", variables=variables),
                f_wwv=nd.parse("h + x + u_co2 ** 2 + aggr(sour(x))", variables=variables),
            ),
            aot_co2_13_30_28=Triplet(
                coeff=nd.parse("1 + sin(t) + cos(t) + sin(5.00e-01 * t) + cos(2.50e-01 * t)", variables=variables),
                f_sst=nd.parse("x + tanh(h) + aggr(sour(h + aggr(sour(h + tanh(h))) * exp(u_co2)))", variables=variables),
                f_wwv=nd.parse("h + u_co2 + sin(u_co2) + aggr(sour(h) + sour(u_aot * x))", variables=variables),
            ),
            co2_320_230_368=Triplet(
                coeff=nd.parse("1 + cos(t) + sin(t) + sin(2 * t) + cos(3 * t) + sin(cos(t)) + sin(5.00e-01 * t)", variables=variables),
                f_sst=nd.parse("x + ENSO * h + tanh(aggr(sour(x))) + tanh(aggr(sour(x))) + aggr(sour(h) * targ(x) + sour(x) + sour(h) * targ(tanh(8.50e-01 * aggr(sour(x))))) + u_co2", variables=variables),
                f_wwv=nd.parse("h + aggr(sour(h)) ** 2 + aggr(sour(h) + sour(x) + targ(h) * tanh(1.25e+00 * sour(x)) + tanh(9.00e-01 * targ(h) * sour(h))) + u_co2", variables=variables),
            ),
            co2_327_240_337=Triplet(
                coeff=nd.parse("1 + cos(t) + sin(t) + sin(2 * t) + cos(3 * t) + sin(0.50 * t) + sin(cos(t))", variables=variables),
                f_sst=nd.parse("u_co2 + x + ENSO * h + tanh(readout(x)) + aggr(sour(h) * targ(x) + sour(h) * tanh(readout(x)) + sour(x))", variables=variables),
                f_wwv=nd.parse("u_co2 + h + aggr(sour(h) + sour(x) + targ(h) * sour(tanh(1.25 * x)))", variables=variables),
            ),
            log_co2_12=Triplet(
                coeff=nd.parse("1 + cos(t) + sin(t) + sin(0.50 * t) + sin(2 * t)", variables=variables),
                f_sst=nd.parse("x + ENSO * h + log(u_co2) + aggr( sour(x) + tanh(0.70 * sour(h)) * (1.00 + 1.20 * targ(x) + 1.10 * readout(x)) )", variables=variables),
                f_wwv=nd.parse("h + readout(x) ** 2       + aggr( sour(x) + tanh(2.00 * sour(h)) * (0.65 + 0.60 * targ(h) + 0.10 * readout(x)) )", variables=variables),
            ),
        )
    def load(self):
        seed_individuals = []
        for name in self.assistant.args.seed_individuals:
            triplet = self.triplet_dict[name]
            indiv = Individual(
                uuid=my_uuid(np.random.default_rng(my_hash(name))),
                n_iter=0,
                source='seed_individuals',
                parents=[],
                triplet=triplet,
                program=self.assistant.triplet_to_program(triplet),
                dynamics=self.assistant.triplet_to_dynamics(triplet),
            )
            seed_individuals.append(indiv)
        for pkl_gz_path in self.assistant.args.load_seed_individuals: #.pkl.gz
            with gzip.open(pkl_gz_path, 'rb') as f:
                individuals = pickle.load(f)
            for indiv_raw in individuals:
                indiv = Individual(
                    uuid=indiv_raw['uuid'],
                    n_iter=0,
                    source=pkl_gz_path,
                    parents=[],                  
                    program=indiv_raw['program'],
                    triplet=Triplet(
                        coeff=nd.parse(indiv_raw['coeff'], variables=self.assistant.variables), 
                        f_sst=nd.parse(indiv_raw['f_sst'], variables=self.assistant.variables), 
                        f_wwv=nd.parse(indiv_raw['f_wwv'], variables=self.assistant.variables),
                    ),
                    dynamics=Dynamics(
                        ddt_sst=nd.parse(indiv_raw['ddt_sst'], variables=self.assistant.variables), 
                        ddt_wwv=nd.parse(indiv_raw['ddt_wwv'], variables=self.assistant.variables),
                    ),
                )
                indiv.detail = {}
                indiv.fitness = None
                indiv.raw_program = indiv_raw.get('raw_program', None)
                seed_individuals.append(indiv)
        return seed_individuals
