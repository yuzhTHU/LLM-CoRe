import time
import logging
import warnings
import nd2py as nd
import numpy as np
import pandas as pd
from functools import reduce
from scipy.optimize import minimize
from scipy.sparse.linalg import lsqr
from typing import Dict, List, Tuple
from scipy.sparse import coo_matrix, csr_matrix, hstack
from ..core import Dynamics
from ..utils import split_ab, df2dict
from ..symbols import RedNoise, PeriodicSpline

_logger = logging.getLogger(__name__)


class ParameterFitter:
    def __init__(self, args, edge_list, num_nodes):
        self.args = args
        self.edge_list = edge_list
        self.num_nodes = num_nodes

    def fit_parameters(self, f: Dynamics, df: pd.DataFrame, nonlinear=True) -> Tuple[Dynamics, float]:
        """ 在 X 上拟合 f 中的参数，返回拟合后的 Dynamics 以及均方误差 """
        A, b_sst, b_wwv = split_ab(df)
        X = df2dict(A)
        sst_items = f.ddt_sst.split_by_add()
        wwv_items = f.ddt_wwv.split_by_add()
        if nonlinear:
            ddt_sst, mse_sst = self.nonlinear_fit(sst_items, X, b_sst.values)
            ddt_wwv, mse_wwv = self.nonlinear_fit(wwv_items, X, b_wwv.values)
        else:
            ddt_sst, mse_sst = self.linear_fit(sst_items, X, b_sst.values)
            ddt_wwv, mse_wwv = self.linear_fit(wwv_items, X, b_wwv.values)
        return Dynamics(ddt_sst, ddt_wwv), mse_sst + mse_wwv

    def linear_fit(self, items: List[nd.Symbol], X: pd.DataFrame, y: pd.DataFrame) -> Tuple[nd.Symbol, float]:
        """Fit a list of items to the data using linear regression."""
        T, M = y.shape  # M = N or 1
        N = self.num_nodes
        E = len(self.edge_list[0]) if self.edge_list is not None else 0
        A_list = []
        eval_args = dict(edge_list=self.edge_list, num_nodes=self.num_nodes)
        for item in items:
            if isinstance(item, nd.Aggr):
                value = item.operands[0].eval(X, **eval_args)
                value = value + np.zeros((T, E))
                assert value.shape == (T, E)
                index = self.edge_list[1]
                cols = np.tile(np.arange(E), (T, 1))
                rows = np.tile(np.arange(T)[:, np.newaxis], (1, E))
                rows += np.array(index) * T
                A = coo_matrix(
                    (value.flatten(), (rows.flatten(), cols.flatten())),
                    shape=(M * T, E),
                ).tocsr()
            elif isinstance(item, nd.Rgga):
                value = item.operands[0].eval(X, **eval_args)
                value = value + np.zeros((T, E))
                assert value.shape == (T, E)
                index = self.edge_list[0]
                cols = np.tile(np.arange(E), (T, 1))
                rows = np.tile(np.arange(T)[:, np.newaxis], (1, E))
                rows += np.array(index) * T
                A = coo_matrix(
                    (value.flatten(), (rows.flatten(), cols.flatten())),
                    shape=(M * T, E),
                ).tocsr()
            elif isinstance(item, nd.Readout) and M == 1:
                value = item.operands[0].eval(X, **eval_args)
                value = value + np.zeros((T, 1))
                assert value.shape in [(T, E), (T, N), (T, 1)]
                A = csr_matrix(np.tile(value, (M, 1))) # (M*T, E)/(M*T, N)/(M*T, 1)
            elif np.size(value := item.eval(X, **eval_args)) == 1:  # (1, 1)
                value = np.eye(M) * value
                value = np.tile(value[:, np.newaxis, :], (1, T, 1)).reshape(M * T, M)
                A = csr_matrix(value) # (M*T, M)
            elif value.ndim == 2 and value.shape[-1] == 1:  # (T, 1)
                assert value.shape == (T, 1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    value = (np.eye(M)[:, np.newaxis, :] * value).reshape(M * T, M)
                A = csr_matrix(value) # (M*T, M)
            else:  # (1, N), (T, N)
                value = value + np.zeros((T, 1))
                assert value.shape == (T, M)
                cols = np.tile(np.arange(M), (T, 1))
                rows = np.tile(np.arange(T)[:, np.newaxis], (1, M))
                rows += np.arange(M) * T
                A = coo_matrix(
                    (value.flatten(), (rows.flatten(), cols.flatten())),
                    shape=(M * T, M),
                ).tocsr()
            A_list.append(A)
        # Linear regression
        A = hstack(A_list, format="csr")
        b = y.T.reshape(M * T)
        if A.shape[0] < A.shape[1] and getattr(self, "__last_A_shape", None) != A.shape:
            _logger.trace(
                f"Linear regression: A.shape={A.shape}, b.shape={b.shape}, items={len(items)}"
            )
            setattr(self, "__last_A_shape", A.shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            vars(self.args).setdefault("damp", 1e-1)
            coeffs, _, _, mse = lsqr(A, b, damp=self.args.damp)[:4]
        # Add coefficients to items
        results = []
        coeff_nums = [x.shape[-1] for x in A_list]
        coeff_list = np.split(coeffs, np.cumsum(coeff_nums)[:-1])
        for item, coeff in zip(items, coeff_list):
            if isinstance(item, nd.Aggr):
                coeff = nd.Number(coeff, nettype="edge")
                result = nd.aggr(coeff * item.operands[0])
            elif isinstance(item, nd.Rgga):
                coeff = nd.Number(coeff, nettype="edge")
                result = nd.rgga(coeff * item.operands[0])
            elif isinstance(item, nd.Readout) and M == 1:
                if np.size(coeff) == 1:
                    coeff = coeff[0]
                coeff = nd.Number(coeff, nettype=item.operands[0].nettype)
                result = nd.readout(coeff * item.operands[0])
            elif np.size(coeff) == 1:
                coeff = nd.Number(coeff[0], nettype="scalar")
                result = coeff * item
            else:
                coeff = nd.Number(coeff, nettype="node")
                result = coeff * item
            results.append(result)
        # Merge items into a single model
        model = reduce(lambda a, b: a + b, results)
        return model, mse

    def nonlinear_fit(self, items: List[nd.Symbol], X: pd.DataFrame, y: pd.DataFrame) -> Tuple[nd.Symbol, float]:
        """Fit the model to the data using least squares, returning the fitted model and the mean squared error (MSE)."""
        start_time = time.time()

        # Collect nonlinear parameters
        items_numbers = [
            item.get_numbers(fitable_only=True, float_only=True)
            for item in items
        ]
        nonlinear_params = [
            [float(number.value) for number in item_numbers]
            for item_numbers in items_numbers
        ]
        params_nums = [len(p) for p in nonlinear_params]
        params_list = sum(nonlinear_params, [])

        def loss(params, return_model=False):
            # Set nonlinear parameters
            params_list = np.split(params, np.cumsum(params_nums)[:-1])
            for item_numbers, params in zip(items_numbers, params_list):
                for number, param in zip(item_numbers, params):
                    number.value = param
            # Fit linear coefficients
            model, mse = self.linear_fit(items, X, y)
            return (model, mse) if return_model else mse

        if len(params_list) and sum(params_nums) < self.args.max_params:
            res = minimize(loss, params_list, method="BFGS", options={"maxiter": 30})
            model, mse = loss(res.x, return_model=True)
        else:
            model, mse = loss(params_list, return_model=True)

        if (duration := time.time() - start_time) > 90:
            _logger.trace(
                f"Taken long time to fit parameters: "
                f"len(items)={len(items)}, "
                f"param_num={sum(params_nums)} "
                f"(BFGS: {len(params_list) and sum(params_nums) < self.args.max_params}), "
                f"time={duration:.0f}s"
            )
        return model, mse

    # def nonlinear_fit(self, items: List[nd.Symbol], X: pd.DataFrame, y: pd.DataFrame) -> Tuple[nd.Symbol, float]:
    #     """Fit the model to the data using least squares, returning the fitted model and the mean squared error (MSE)."""
    #     start_time = time.time()

    #     # Collect nonlinear parameters
    #     param_dict = {}
    #     for item in items:
    #         for number in item.get_numbers(fitable_only=True, float_only=True):
    #             if number.name not in param_dict:
    #                 param_dict[number.name] = []
    #             param_dict[number.name].append(number)
    #     params_list = list(param_dict.values())
    #     params_nums = len(param_dict)

    #     def loss(params, return_model=False):
    #         for param, param_list in zip(params, params_list):
    #             for number in param_list:
    #                 number.value = param
    #         # Fit linear coefficients
    #         model, mse = self.linear_fit(items, X, y)
    #         return (model, mse) if return_model else mse

    #     x0 = [param_list[0].value for param_list in params_list]
    #     if len(x0) and (params_nums) < self.args.max_params:
    #         res = minimize(loss, x0, method="BFGS", options={"maxiter": 30})
    #         model, mse = loss(res.x, return_model=True)
    #     else:
    #         model, mse = loss(x0, return_model=True)

    #     if (duration := time.time() - start_time) > 90:
    #         _logger.trace(
    #             f"Taken long time to fit parameters: "
    #             f"len(items)={len(items)}, "
    #             f"param_num={(params_nums)} "
    #             f"(BFGS: {len(x0) and (params_nums) < self.args.max_params}), "
    #             f"time={duration:.0f}s"
    #         )
    #     return model, mse

    def fit_red_noise(self, f: Dynamics, df: pd.DataFrame, n_member=1, n=1, max_lag=2, random_state=42) -> Dynamics:
        """ 在 X 上拟合 f 中的红噪声参数，返回添加了红噪声项的 Dynamics 
        Args:
        - n: 由于红噪声是离散随机过程, 因此需要指定 n, 表示一个 dt 时间步内的子步数
        - max_lag: 用于拟合红噪声的最大滞后阶数
        - random_state: 随机数种子
        """
        ## 计算单步拟合的残差
        A, b_sst, b_wwv = split_ab(df)
        X = df2dict(A)
        f_sst = f.ddt_sst.eval(X, edge_list=self.edge_list, num_nodes=self.num_nodes)
        f_wwv = f.ddt_wwv.eval(X, edge_list=self.edge_list, num_nodes=self.num_nodes)
        residual_sst = b_sst - f_sst
        residual_wwv = b_wwv - f_wwv
        nettype_sst = {1: 'scalar', self.num_nodes: 'node', len(self.edge_list[0]): 'edge'}.get(len(df['x'].columns))
        nettype_wwv = {1: 'scalar', self.num_nodes: 'node', len(self.edge_list[0]): 'edge'}.get(len(df['h'].columns))
        ## 用单步残差获取红噪声
        a_sst = self.fit_a(residual_sst, max_lag=max_lag, n=n)
        a_wwv = self.fit_a(residual_wwv, max_lag=max_lag, n=n)
        shape_sst = [n_member, *df['x'].iloc[0].shape]
        shape_wwv = [n_member, *df['h'].iloc[0].shape]
        noise_sst = RedNoise(a=a_sst.values, shape=shape_sst, nettype=nettype_sst, random_state=random_state)
        noise_wwv = RedNoise(a=a_wwv.values, shape=shape_wwv, nettype=nettype_wwv, random_state=random_state)
        ## 用单步残差获取季节性标准差
        t = nd.Variable("t", nettype="scalar")
        stdac_sst = self.fit_stdac(residual_sst)
        stdac_wwv = self.fit_stdac(residual_wwv)
        std_sst = PeriodicSpline(t, x=A['t'].values[:13, 0], y=stdac_sst.values, nettype=nettype_sst)
        std_wwv = PeriodicSpline(t, x=A['t'].values[:13, 0], y=stdac_wwv.values, nettype=nettype_wwv)
        ## 加入红噪声项
        noisy_ddt_sst = f.ddt_sst.copy() + std_sst * noise_sst
        noisy_ddt_wwv = f.ddt_wwv.copy() + std_wwv * noise_wwv
        return Dynamics(noisy_ddt_sst, noisy_ddt_wwv)

    def fit_a(self, residuals: pd.DataFrame, max_lag: int, n: int) -> pd.DataFrame:
        ## 计算各模态的红噪声系数 a1 (mode,)
        with np.errstate(invalid="ignore", divide="ignore"):
            x = residuals - residuals.mean()
            var = x.var(ddof=0)
            a_dict = {}
            for lag in range(1, max_lag+1): # 红噪声: corr(lag)/corr(0) ~ a ^ lag
                corr = (x * x.shift(-lag)).mean()
                a_dict[lag] = (corr / var) ** (1 / lag)
            a = pd.DataFrame(a_dict).mean(axis='columns').to_frame().T
            a = np.sign(a) * np.abs(a) ** (1 / n)  # Adjust for the number of steps
            return a  # (1, len(residuals))

    def fit_stdac(self, residuals: pd.DataFrame):
        ## 计算各模态的季节性标准差 stdac (month, mode)
        with np.errstate(invalid="ignore", divide="ignore"):
            stdac = (
                residuals.assign(month=residuals.index.month)
                .groupby("month").std()
                .loc[residuals.index.month.values[:12]]
            )
            return stdac  # (12, len(residuals))


class ParameterFitterv2(ParameterFitter):
    def nonlinear_fit(self, items: List[nd.Symbol], X: pd.DataFrame, y: pd.DataFrame) -> Tuple[nd.Symbol, float]:
        """Fit the model to the data using least squares, returning the fitted model and the mean squared error (MSE)."""
        start_time = time.time()

        # Collect nonlinear parameters
        nonlinear_params = {} # {value: [Number1, Number2, ...]}
        for item in items:
            for number in item.get_numbers(fitable_only=True, float_only=True, scalar_only=True):
                value = float(number.value)
                nonlinear_params[value] = nonlinear_params.get(value, []) + [number]
        params_num = len(nonlinear_params)
        params_list = list(nonlinear_params.keys())
        numbers_list = list(nonlinear_params.values())

        def loss(params, return_model=False):
            # Set nonlinear parameters
            for numbers, param in zip(numbers_list, params):
                for number in numbers:
                    number.value = param
            # Fit linear coefficients
            model, mse = self.linear_fit(items, X, y)
            return (model, mse) if return_model else mse

        if 0 < params_num <= self.args.max_params:
            res = minimize(loss, params_list, method="BFGS", options={"maxiter": 30})
            model, mse = loss(res.x, return_model=True)
        else:
            model, mse = loss(params_list, return_model=True)

        if (duration := time.time() - start_time) > 90:
            _logger.trace(
                f"Taken long time to fit parameters: "
                f"len(items)={len(items)}, "
                f"param_num={params_num} "
                f"(BFGS: {'res' in locals()}), "
                f"time={duration:.0f}s"
            )
        return model, mse