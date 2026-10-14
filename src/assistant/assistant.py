import logging
import nd2py as nd
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from ..core import Triplet, Dynamics, Individual
from .data_loader import DataLoader
from .frozen_base import FrozenMixin
from .indiv2prompt import Indiv2Prompt
from .triplet2program import Triplet2Program
from .program2triplet import Program2Triplet
from .markdown2program import Markdown2Program
from .triplet2dynamics import Triplet2Dynamics
from .triplet2reduction import Triplet2Reduction
from .seed_individuals_loader import SeedIndividualsLoader
from .baseline_indiv_loader import BaselineIndivLoader

_logger = logging.getLogger(__name__)


class Assistant(FrozenMixin):
    def __init__(self, args):
        self.args = args
        ## Setup modes
        self.modes = ['ENSO', 'NPMM', 'SPMM', 'IOB', 'IOD', 'SIOD', 'TNA', 'ATL3', 'SASD']
        ## Setup variable dict
        basic_variables = {
            'x': nd.Variable('x', nettype='node'),
            'h': nd.Variable('h', nettype='node'),
            't': nd.Variable('t', nettype='scalar'),
        }
        mode_variables = {
            'ENSO': nd.Variable('ENSO', nettype='node'),
            'NPMM': nd.Variable('NPMM', nettype='node'),
            'ENSO': nd.Variable("ENSO", nettype="node"),
            'NPMM': nd.Variable("NPMM", nettype="node"),
            'SPMM': nd.Variable("SPMM", nettype="node"),
            'IOB':  nd.Variable("IOB", nettype="node"),
            'IOD':  nd.Variable("IOD", nettype="node"),
            'SIOD': nd.Variable("SIOD", nettype="node"),
            'TNA':  nd.Variable("TNA", nettype="node"),
            'ATL3': nd.Variable("ATL3", nettype="node"),
            'SASD': nd.Variable("SASD", nettype="node"),
        }
        forcing_variables = {
            'u_co2': nd.Variable('u_co2', nettype='scalar'),
            'u_aot': nd.Variable('u_aot', nettype='scalar'),
            'u_tsi': nd.Variable('u_tsi', nettype='scalar'),
            'u_temp': nd.Variable('u_temp', nettype='scalar'),
        }
        if args.setting == 'default':
            self.variables = {**basic_variables, **mode_variables, **forcing_variables}
        elif args.setting == 'global_wwv':
            basic_variables['h'].nettype = 'scalar'
            self.variables = {**basic_variables, **mode_variables, **forcing_variables}
        elif args.setting == 'no_forcing':
            forcing_variables = {}
            self.variables = {**basic_variables, **mode_variables, **forcing_variables}
        elif args.setting == 'only_co2':
            forcing_variables = {'u_co2': forcing_variables['u_co2']}
            self.variables = {**basic_variables, **mode_variables, **forcing_variables}
        elif args.setting == 'only_aot':
            forcing_variables = {'u_aot': forcing_variables['u_aot']}
            self.variables = {**basic_variables, **mode_variables, **forcing_variables}
        elif args.setting == 'only_tsi':
            forcing_variables = {'u_tsi': forcing_variables['u_tsi']}
            self.variables = {**basic_variables, **mode_variables, **forcing_variables}
        elif args.setting == 'only_temp':
            forcing_variables = {'u_temp': forcing_variables['u_temp']}
            self.variables = {**basic_variables, **mode_variables, **forcing_variables}
        elif args.setting == 'aot_and_tsi':
            forcing_variables = {'u_aot': forcing_variables['u_aot'], 'u_tsi': forcing_variables['u_tsi']}
            self.variables = {**basic_variables, **mode_variables, **forcing_variables}
        elif args.setting == 'aot_and_co2':
            forcing_variables = {'u_aot': forcing_variables['u_aot'], 'u_co2': forcing_variables['u_co2']}
            self.variables = {**basic_variables, **mode_variables, **forcing_variables}
        elif args.setting == 'homogeneous':
            self.variables = {**basic_variables, **forcing_variables}
        elif args.setting == 'homo_co2':
            forcing_variables = {'u_co2': forcing_variables['u_co2']}
            self.variables = {**basic_variables, **forcing_variables}
        elif args.setting == 'homo_aot_and_co2':
            forcing_variables = {'u_aot': forcing_variables['u_aot'], 'u_co2': forcing_variables['u_co2']}
            self.variables = {**basic_variables, **forcing_variables}
        else:
            raise ValueError(f"Unknown setting: {args.setting}")
        ## Setup signature
        if args.setting == "default":
            self.signature = "def get_equations(t, x, h, u_co2, u_aot, u_tsi, u_temp, mode, nl_params):"
        elif args.setting == "global_wwv":
            self.signature = "def get_equations(t, x, h, u_co2, u_aot, u_tsi, u_temp, mode, nl_params):"
        elif args.setting == 'no_forcing':
            self.signature = "def get_equations(t, x, h, mode, nl_params):"
        elif args.setting == 'only_co2':
            self.signature = "def get_equations(t, x, h, u_co2, mode, nl_params):"
        elif args.setting == 'only_aot':
            self.signature = "def get_equations(t, x, h, u_aot, mode, nl_params):"
        elif args.setting == 'only_tsi':
            self.signature = "def get_equations(t, x, h, u_tsi, mode, nl_params):"
        elif args.setting == 'only_temp':
            self.signature = "def get_equations(t, x, h, u_temp, mode, nl_params):"
        elif args.setting == 'aot_and_tsi':
            self.signature = "def get_equations(t, x, h, u_aot, u_tsi, mode, nl_params):"
        elif args.setting == 'aot_and_co2':
            self.signature = "def get_equations(t, x, h, u_co2, u_aot, mode, nl_params):"
        elif args.setting == 'homogeneous':
            self.signature = "def get_equations(t, x, h, u_co2, u_aot, u_tsi, u_temp, nl_params):"
        elif args.setting == 'homo_co2':
            self.signature = "def get_equations(t, x, h, u_co2, nl_params):"
        elif args.setting == 'homo_aot_and_co2':
            self.signature = "def get_equations(t, x, h, u_co2, u_aot, nl_params):"
        else:
            raise ValueError(f"Unknown prompt: {args.prompt}")
        ## Setup signature_args
        nl_params = np.random.rand(100)
        if args.setting == "default":
            self.signature_args = {**basic_variables, **forcing_variables, 'mode': mode_variables, 'nl_params': nl_params}
        elif args.setting == "global_wwv":
            self.signature_args = {**basic_variables, **forcing_variables, 'mode': mode_variables, 'nl_params': nl_params}
        elif args.setting == 'no_forcing':
            self.signature_args = {**basic_variables, **forcing_variables, 'mode': mode_variables, 'nl_params': nl_params}
        elif args.setting == 'only_co2':
            self.signature_args = {**basic_variables, **forcing_variables, 'mode': mode_variables, 'nl_params': nl_params}
        elif args.setting == 'only_aot':
            self.signature_args = {**basic_variables, **forcing_variables, 'mode': mode_variables, 'nl_params': nl_params}
        elif args.setting == 'only_tsi':
            self.signature_args = {**basic_variables, **forcing_variables, 'mode': mode_variables, 'nl_params': nl_params}
        elif args.setting == 'only_temp':
            self.signature_args = {**basic_variables, **forcing_variables, 'mode': mode_variables, 'nl_params': nl_params}
        elif args.setting == 'aot_and_tsi':
            self.signature_args = {**basic_variables, **forcing_variables, 'mode': mode_variables, 'nl_params': nl_params}
        elif args.setting == 'aot_and_co2':
            self.signature_args = {**basic_variables, **forcing_variables, 'mode': mode_variables, 'nl_params': nl_params}
        elif args.setting == 'homogeneous':
            self.signature_args = {**basic_variables, **forcing_variables, 'nl_params': nl_params}
        elif args.setting == 'homo_co2':
            self.signature_args = {**basic_variables, **forcing_variables, 'nl_params': nl_params}
        elif args.setting == 'homo_aot_and_co2':
            self.signature_args = {**basic_variables, **forcing_variables, 'nl_params': nl_params}
        else:
            raise ValueError(f"Unknown prompt: {args.prompt}")
        ## Setup namespace
        self.namespace = {
            "add": nd.Add, "sub": nd.Sub, "mul": nd.Mul, "div": nd.Div, "pow": nd.Pow,
            "max": nd.Max, "min": nd.Min, "sin": nd.Sin, "cos": nd.Cos, "tan": nd.Tan,
            "sec": nd.Sec, "csc": nd.Csc, "cot": nd.Cot, "log": nd.Log, "logabs": nd.LogAbs,
            "exp": nd.Exp, "abs": nd.Abs, "neg": nd.Neg, "inv": nd.Inv, "sqrt": nd.Sqrt,
            "arcsin": nd.Arcsin, "arccos": nd.Arccos, "arctan": nd.Arctan, 
            "sinh": nd.Sinh, "cosh": nd.Cosh, "tanh": nd.Tanh, "sech": nd.Sech, 
            "csch": nd.Csch, "sqrtabs": nd.SqrtAbs, "pow2": nd.Pow2, "pow3": nd.Pow3, 
            "sigmoid": nd.Sigmoid, "regular": nd.Regular, "readout": nd.Readout,
            "sour": nd.Sour, "targ": nd.Targ, "aggr": nd.Aggr, "rgga": nd.Rgga,
            "a": 1, "b": 1, "c": 1, "d": 1, "e": 1, "f": 1, "pi": 1,
            "sigmoidal": nd.Sigmoid, 'sigm': nd.Sigmoid, 'sig': nd.Sigmoid,
            "Variable": nd.Variable,
            # "Constant": nd.Constant,
        }
        ## Setup task description
        prompt_dir = Path(__file__).parent.parent / 'prompts'
        if args.setting == 'default':
            self.task_description = (prompt_dir / "default.md").read_text()
        elif args.setting == 'global_wwv':
            self.task_description = (prompt_dir / "global_wwv.md").read_text()
        elif args.setting == 'no_forcing':
            self.task_description = (prompt_dir / "no_forcing.md").read_text()
        elif args.setting == 'only_co2':
            self.task_description = (prompt_dir / "only_co2.md").read_text()
        elif args.setting == 'only_aot':
            self.task_description = (prompt_dir / "only_aot.md").read_text()
        elif args.setting == 'only_tsi':
            self.task_description = (prompt_dir / "only_tsi.md").read_text()
        elif args.setting == 'only_temp':
            self.task_description = (prompt_dir / "only_temp.md").read_text()
        elif args.setting == 'aot_and_tsi':
            self.task_description = (prompt_dir / "aot_and_tsi.md").read_text()
        elif args.setting == 'aot_and_co2':
            self.task_description = (prompt_dir / "aot_and_co2.md").read_text()
        elif args.setting == 'homogeneous':
            self.task_description = (prompt_dir / "homogeneous.md").read_text()
        elif args.setting == 'homo_co2':
            self.task_description = (prompt_dir  /'homo_co2.md').read_text()
        elif args.setting == 'homo_aot_and_co2':
            self.task_description = (prompt_dir  /'homo_aot_and_co2.md').read_text()
        else:
            raise ValueError(f"Unknown setting: {args.setting}")
    
    def load_variables(self) -> Dict[str, nd.Symbol]:
        """ 加载与 setting 对应的变量字典 """
        return self.variables

    def load_data(self, dropna=True, name='default', rescale_ohc=True, normalize_co2=True, normalize_aot=True, normalize_tsi=True) -> pd.DataFrame:
        """ 加载包含所有变量的数据, 抛弃缺失了所需变量的行, 保证可以用于评估:
        - 生成的公式
        - seed individuals
        - baseline indiv 
        """
        df_all = DataLoader.create(self, name).load(rescale_ohc=rescale_ohc, normalize_co2=normalize_co2, normalize_aot=normalize_aot, normalize_tsi=normalize_tsi)
        if dropna:
            valid_idx = df_all[self.variables.keys()].dropna().index
            df_all = df_all.loc[valid_idx].sort_index()
        # 确保 df_all index 连续
        full_range = pd.period_range(start=df_all.index.min(), end=df_all.index.max(), freq='M')
        if len(missing_months := full_range.difference(df_all.index)) > 0:
            raise RuntimeError(f"DataFrame index has missing months: {missing_months}")
        return df_all
    
    def load_topology(self, remove_self_loop=True) -> Tuple[np.ndarray, int]:
        """ 加载与 modes 相对应的 edge_list 和 num_nodes """
        num_nodes = len(self.modes)
        adj = np.ones((num_nodes, num_nodes))
        if remove_self_loop:
            adj = adj - np.eye(num_nodes)
        edge_list = np.stack(np.nonzero(adj), axis=0)
        return edge_list, num_nodes
    
    def load_seed_individuals(self) -> List[Individual]:
        """ 加载未拟合的 seed_individuals """
        seed_individuals = SeedIndividualsLoader.create(self).load()
        return seed_individuals
    
    def load_baseline_indiv(self) -> Individual:
        """ 加载未拟合的 baseline_indiv """
        baseline_indiv = BaselineIndivLoader.create(self).load()
        return baseline_indiv

    def triplet_to_program(self, triplet: Triplet) -> str:
        """ 将 triplet 三元组转换成对应的 program 代码, 除常数外完全保证可逆 """
        tool = Triplet2Program.create(self)
        program = tool(triplet)
        return program
    
    def triplet_to_dynamics(self, triplet: Triplet) -> Dynamics:
        """ 将 triplet 三元组转换成 Dynamics 对象, 完全保证可逆 """
        tool = Triplet2Dynamics.create(self)
        dynamics = tool(triplet)
        return dynamics
    
    def triplet_to_reduction(self, triplet: Triplet) -> Triplet:
        """ 将太复杂的 triplet 三元组进行简化, 不需要可逆 """
        tool = Triplet2Reduction.create(self)
        reduction = tool(triplet)
        return reduction

    def markdown_to_program(self, markdown: str) -> str:
        """ 尝试从 markdown 中提取有效的 (语法正确的) 函数定义部分 """
        tool = Markdown2Program.create(self)
        program = tool(markdown)
        return program
    
    def program_to_triplet(self, program: str) -> Triplet:
        """ 将 program 代码转换成对应的 triplet 三元组, 需要大致保证可逆 """
        tool = Program2Triplet.create(self)
        triplet = tool(program)
        return triplet

    def ask_for_generate(self, individuals: List[Individual]) -> List[Dict[str, str]]:
        tool = Indiv2Prompt.create(self)
        messages = tool.ask_for_generate(individuals)
        return messages

    def ask_for_explain(self, indiv: Individual) -> List[Dict[str, str]]:
        tool = Indiv2Prompt.create(self)
        messages = tool.ask_for_explain(indiv)
        return messages
    
    def summary_performance(self, indiv: Individual) -> str:
        tool = Indiv2Prompt.create(self)
        performance = tool.summary_performance(indiv)
        return performance