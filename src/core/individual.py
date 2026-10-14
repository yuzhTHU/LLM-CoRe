import pandas as pd
from typing import List, Dict
from .triplet import Triplet
from .dynamics import Dynamics
from ..utils import jsonalize


class Individual:    
    def __init__(
        self,
        uuid: str,
        n_iter: int,
        source: str,
        parents: List[str],
        program: str,
        triplet: Triplet,
        dynamics: Dynamics,
    ):
        """个体，包含唯一标识符、简化的演化动力学、合并的演化动力学、详细指标、适应度、溯源信息
        Attrs:
            uuid (str): 个体唯一标识符
            n_iter (int): 个体出生的迭代轮次
            source (str): 个体来源（如随机初始化、变异、交叉等）
            parents (List[str]): 父代个体的UUID列表
            triplet (Triplet): 拆分的演化动力学 (coeff, f_sst, f_wwv)
            dynamics (Dynamics): 合并的演化动力学对象 (与 coeff, f_sst, f_wwv 完全对应)
            detail (Dict): 详细指标字典
            fitness (float): 适应度值
            program (str): 动力学对应的代码 (与 coeff, f_sst, f_wwv 完全对应)
            raw_program (str): 原始生成程序代码 (只对 LLM 生成的个体有效)
            explanation (str): 对该个体的解释
        """
        self.uuid: str = uuid
        self.n_iter: int = n_iter
        self.source: str = source
        self.parents: List[str] = parents
        self.program: str = program
        self.triplet: Triplet = triplet
        self.dynamics: Dynamics = dynamics
        
        self.detail: Dict = None
        self.fitness: float = None
        self.raw_program: str = None
        self.explanation: str = None
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"Individual("
            f"uuid={self.uuid}, "
            f"n_iter={self.n_iter}, "
            f"source={self.source}, "
            f"triplet={self.triplet}, "
            f"fitness={self.fitness})"
        )

    def json_dict(self) -> dict:
        """Convert the individual to a JSON-serializable dictionary."""
        detail_json = {}
        for key, value in self.detail.items():
            if key in ['GENERATION', 'PREDICTION']:
                continue
            elif isinstance(value, float):
                value = round(value, 4)
            elif isinstance(value, pd.Series):
                if value.index.nlevels > 1:
                    value = value.copy()
                    value.index = [':'.join(map(str, levels)) for levels in value.index]
                value = value.round(4).to_dict()
            detail_json[key] = value

        record = {
            "uuid": self.uuid,
            'n_iter': self.n_iter,
            'source': self.source,
            'parents': self.parents,
            "coeff": self.triplet.coeff.to_str(raw=True),
            "f_sst": self.triplet.f_sst.to_str(raw=True),
            "f_wwv": self.triplet.f_wwv.to_str(raw=True),
            "ddt_sst": self.dynamics.ddt_sst.to_str(raw=True),
            "ddt_wwv": self.dynamics.ddt_wwv.to_str(raw=True),
            **detail_json,
            "fitness": self.fitness,
            "program": self.program,
            "raw_program": self.raw_program,
            "explanation": self.explanation,
        }
        return record