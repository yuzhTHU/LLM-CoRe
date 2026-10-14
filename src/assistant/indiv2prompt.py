import ast
import logging
import textwrap
import nd2py as nd
from typing import List, Dict
from abc import ABC, abstractmethod
from ..core import Individual
from .frozen_base import FrozenMixin

_logger = logging.getLogger(__name__)


class Indiv2Prompt(FrozenMixin, ABC):
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
    def create(cls, assistant) -> 'Indiv2Prompt':
        sub_class = cls.SUBCLASS_REGISTRY.get("default")
        return sub_class(assistant)

    @abstractmethod
    def ask_for_generate(self, individuals: List[Individual]) -> List[Dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def ask_for_explain(self, indiv: Individual) -> List[Dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def summary_performance(self, indiv: Individual) -> str:
        raise NotImplementedError


@Indiv2Prompt.register('default')
class Indiv2Prompt_v0(Indiv2Prompt):
    def ask_for_generate(self, individuals: List[Individual]) -> List[Dict[str, str]]:
        messages = []
        ## 系统给出任务说明
        messages.append({'role': 'system', 'content': self.assistant.task_description})
        for idx, indiv in enumerate(sorted(individuals, key=lambda x: x.fitness)):
            ## LLM 应用户要求生成 program
            what = 'a program' if idx == 0 else 'a better program based on the results above'
            messages.append({'role': 'user', 'content': f'Generate {what}.'})
            messages.append({'role': 'assistant', 'content': indiv.program})
            ## 用户给出对 LLM 生成结果的评价
            how = 'good' if idx == 0 else 'better than the previous one'
            messages.append({'role': 'user', 'content': (
                f'This program is {how}, with fitness={indiv.fitness:.4g}. '
                f'Detailed evaluation:\n{self.summary_performance(indiv)}'
            )})
            ## LLM 应用户要求生成 explanation
            if indiv.explanation is not None:
                messages.append({'role': 'user', 'content': 'Explain it.'})
                messages.append({'role': 'assistant', 'content': indiv.explanation})
        ## 用户要求 LLM 生成新的 program
        messages.append({'role': 'user', 'content': 'Generate a better program based on the results above.'})
        return messages

    def ask_for_explain(self, indiv: Individual) -> List[Dict[str, str]]:
        messages = [
            ## 系统给出任务说明
            {'role': 'system', 'content': self.assistant.task_description},
            ## LLM 应用户要求生成 program
            {'role': 'user', 'content': 'Generate a program.'},
            {'role': 'assistant', 'content': indiv.program},
            ## 用户给出对 LLM 生成结果的评价
            {'role': 'user', 'content': (
                f'This program is good, with fitness={indiv.fitness:.4g}. '
                f'Detailed evaluation:\n{self.summary_performance(indiv)}'
            )},
            ## 用户要求 LLM 生成 explanation
            {'role': 'user', 'content': (
                f"Explain the program you generated in detail, such as the overall structure, "
                f"the purpose of each part / term, and why it can help to improve the fitness."
            )},
        ]
        return messages

    def summary_performance(self, indiv: Individual) -> str:
        performance = (
            f"- Prediction Skills: " + ", ".join(f"{a}_{b}={v} months" for (a, b), v in indiv.detail['skill'].items()) + ";\n"
            f"- Prediction Skills Improvement (compared to XRO): " + ", ".join(f"{a}_{b}={v:.2%}" for (a, b), v in indiv.detail['skill%'].items()) + ";\n"
            f"- Trajectory RMSE: {indiv.detail["traj_rmse"].mean():.4g}\n"
            f"- Formula Complexity: {indiv.detail["length"]}\n"
            f"- ENSO Monthly Standard Deviation: " + ", ".join(f"{a}={v:.2g}" for a, v in indiv.detail['std'].items()) +
            f" (MAE with Ground Truth: {indiv.detail["std_mae"]:.4g});\n"
            f"- ENSO Monthly Skewness: " + ", ".join(f"{a}={v:.2g}" for a, v in indiv.detail['skewness'].items()) +
            f" (MAE with Ground Truth: {indiv.detail["skewness_mae"]:.4g});\n"
            f"- ENSO Spectrum Density: " + ", ".join(f"{a}={v:.2g}" for a, v in indiv.detail['spectrum'].items()) +
            f" (MAE with Ground Truth: {indiv.detail["spectrum_mae"]:.4g});\n"
            f"- Lead-Lag Correlation MAE with Ground Truth: {indiv.detail["lead_lag_mae"]:.4g}\n"
            f"- Forecast Lead Month: " + ", ".join(f"{a}={v:.2g} months" for a, v in indiv.detail['forecast'].items()) + ";\n"
        )
        return performance
