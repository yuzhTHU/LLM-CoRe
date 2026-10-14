import logging
import warnings
import pandas as pd
from copy import deepcopy
from tqdm.rich import tqdm
from ..core import Dynamics
from ..utils import Intergrator, df2dict

_logger = logging.getLogger(__name__)


class RolloutSimulator:
    def __init__(self, args, edge_list, num_nodes):
        self.args = args
        self.edge_list = edge_list
        self.num_nodes = num_nodes
        self.intergrator = Intergrator.load(args.intergrator)
        self.clip_range = None

    def set_clip_bounds(self, df: pd.DataFrame, clip_factor: float):
        q1, q3 = df.quantile(0.25), df.quantile(0.75)
        iqr = q3 - q1
        upper = q3 + clip_factor * iqr
        lower = q1 - clip_factor * iqr
        self.clip_range = pd.DataFrame({'upper': upper, 'lower': lower}).T

    def clip_X(self, X: dict) -> dict:
        if self.clip_range is None:
            raise ValueError("Clip bounds not set. Call set_clip_bounds first.")
        for idx, (mode, v) in enumerate(self.clip_range['x'].items()):
            X['x'][:, idx] = X['x'][:, idx].clip(v.lower, v.upper)
        for idx, (mode, v) in enumerate(self.clip_range['h'].items()):
            X['h'][:, idx] = X['h'][:, idx].clip(v.lower, v.upper)
        return X

    def get_simulation(
            self, f: Dynamics, df: pd.DataFrame, dt: float, tqdm_desc=None, max_month=None,
        ) -> pd.DataFrame:
        """ Simulate f on df for several months
        Args: df (pd.DataFrame) 用于模拟的初始状态
            df 包含两层 column (MultiIndex), 外层包含 t, sst, wwv 列以及若干强迫变量列, 内层为气候模态名称
            在 df 中包含至少两帧时, dt 可以根据 df['t'] 自动推断; 否则需要手动指定 dt
        Returns: simulation (pd.DataFrame) 从各帧 (行) 开始模拟 lead 月份的结果
            simulation 在 df 的基础上增加一层 index (raw_index, lead), 表示初始时间和预测月份
        """
        _logger.critical(f"This function has been deprecated. Please use `get_simulation_with_forcing` instead, which allows you to specify the forcing variables during simulation.")
        if max_month is None: 
            max_month = self.args.max_month
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ## 存储当前状态
            X = df2dict(df, keep_shape=False)
            ## 记录模拟结果
            df_list = [df.copy().assign(lead=0)]  # .copy() 解决碎片化问题
            ## 开始模拟
            for lead in tqdm(range(1, max_month + 1), disable=tqdm_desc is None, desc=tqdm_desc):
                ## 存储子步平均结果
                X_avg = {'t': 0 * X['t'], 'x': 0 * X['x'], 'h': 0 * X['h']}
                ## 子步模拟
                for _ in range(self.args.n_steps):
                    dsst, dwwv = self.intergrator.step(
                        f, X, dt / self.args.n_steps, edge_list=self.edge_list, num_nodes=self.num_nodes
                    )
                    X["x"] = X["x"] + dsst
                    X["h"] = X["h"] + dwwv
                    ## 更新 t 的两种方案：子步内保持不变 or 子步内保持更新
                    if self.args.update_t_within_step: 
                        X['t'] += dt / self.args.n_steps
                    ## 剪裁异常状态
                    if self.clip_range is not None:
                        X = self.clip_X(X)
                    ## 累加子步结果
                    X_avg['t'] = X_avg['t'] + X['t']
                    X_avg["x"] = X_avg["x"] + X["x"]
                    X_avg["h"] = X_avg["h"] + X["h"]
                if not self.args.update_t_within_step:
                    X["t"] += dt
                ## 将
                df_avg = df.copy().assign(lead=lead)
                df_avg.loc[:, 't'] = X_avg['t'] / self.args.n_steps
                df_avg.loc[:, 'x'] = X_avg['x'] / self.args.n_steps
                df_avg.loc[:, 'h'] = X_avg['h'] / self.args.n_steps
                df_list.append(df_avg)
        simulation = pd.concat(df_list, axis=0).set_index('lead', append=True).sort_index()
        simulation = simulation.loc[:, ['t', 'x', 'h']]
        return simulation

    def get_simulation_with_forcing(
            self, f: Dynamics, df: pd.DataFrame, dt: float, df_forcing: pd.DataFrame, tqdm_desc=None, max_month=None,
        ) -> pd.DataFrame:
        """ Simulate f on df for several months
        Args: df (pd.DataFrame) 用于模拟的初始状态
            df 包含两层 column (MultiIndex), 外层包含要模拟的 x, h 列, 内层为气候模态名称
            df_forcing 同样包含两层 column (MultiIndex), 包含若干强迫变量列
            在 df_forcing 中包含至少两帧时, dt 可以根据 df['t'] 自动推断; 否则需要手动指定 dt
        Returns: simulation (pd.DataFrame) 从各帧 (行) 开始模拟 lead 月份的结果
            simulation 在 df 的基础上增加一层 index (raw_index, lead), 表示初始时间和预测月份
        """
        if max_month is None: 
            max_month = self.args.max_month
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ## 存储当前状态
            X = df2dict(df, keep_shape=True)
            df_idx = df_forcing.index.get_indexer(df.index)
            ## 记录模拟结果
            df_list = [df.copy().assign(lead=0)]  # .copy() 解决碎片化问题
            ## 开始模拟
            for lead in tqdm(range(1, max_month + 1), disable=tqdm_desc is None, desc=tqdm_desc):
                X.update(df2dict(df_forcing.iloc[df_idx], keep_shape=False))
                ## 存储子步平均结果
                X_avg = {'t': 0 * X['t'], 'x': 0 * X['x'], 'h': 0 * X['h']}
                ## 子步模拟
                for _ in range(self.args.n_steps):
                    dsst, dwwv = self.intergrator.step(
                        f, X, dt / self.args.n_steps, edge_list=self.edge_list, num_nodes=self.num_nodes
                    )
                    X["x"] = X["x"] + dsst
                    X["h"] = X["h"] + dwwv
                    ## 更新 t 的两种方案：子步内保持不变 or 子步内保持更新
                    if self.args.update_t_within_step: 
                        X['t'] += dt / self.args.n_steps
                    ## 剪裁异常状态
                    if self.clip_range is not None:
                        X = self.clip_X(X)
                    ## 累加子步结果
                    X_avg['t'] = X_avg['t'] + X['t']
                    X_avg["x"] = X_avg["x"] + X["x"]
                    X_avg["h"] = X_avg["h"] + X["h"]
                if not self.args.update_t_within_step:
                    X["t"] += dt
                ## 将平均结果赋值给 df_avg
                df_avg = df.copy().assign(lead=lead)
                df_avg.loc[:, 't'] = X_avg['t'] / self.args.n_steps
                df_avg.loc[:, 'x'] = X_avg['x'] / self.args.n_steps
                df_avg.loc[:, 'h'] = X_avg['h'] / self.args.n_steps
                df_list.append(df_avg)
                df_idx = (df_idx + 1).clip(None, len(df_forcing) - 1)
        simulation = pd.concat(df_list, axis=0).set_index('lead', append=True).sort_index()
        simulation = simulation.loc[:, ['t', 'x', 'h']]
        return simulation
