import numpy as np
from typing import Dict
from copy import deepcopy
from ..core import Dynamics


class Intergrator:
    @staticmethod
    def load(name: str, *args, **kwargs) -> 'Intergrator':
        name = name.lower()
        if name == "euler":
            return EulerIntergrator(*args, **kwargs)
        elif name == "midpoint":
            return MidpointIntegrator(*args, **kwargs)
        elif name == "trapezoidal":
            return TrapezoidalIntegrator(*args, **kwargs)
        elif name == "rk4":
            return RK4Integrator(*args, **kwargs)
        else:
            raise ValueError(f"Unknown integrator: {name}")

    def step(self, f: Dynamics, X: Dict[str, np.ndarray], dt, **eval_args):
        raise NotImplementedError


class EulerIntergrator(Intergrator):
    """显式欧拉法（不可逆）"""
    def step(self, f: Dynamics, X: Dict[str, np.ndarray], dt, **eval_args):
        dsst = f.ddt_sst.eval(X, **eval_args) * dt
        dwwv = f.ddt_wwv.eval(X, **eval_args) * dt
        return dsst, dwwv


class MidpointIntegrator(Intergrator):
    """中点法（时间可逆、显式二阶）"""
    def step(self, f: Dynamics, X: Dict[str, np.ndarray], dt, **eval_args):
        # Step 1: 基于当前状态计算导数
        dsst_0 = f.ddt_sst.eval(X, **eval_args) * dt
        dwwv_0 = f.ddt_wwv.eval(X, **eval_args) * dt

        # Step 2: 构造中点预测状态
        X_mid = deepcopy(X)
        X_mid["x"] = X["x"] + 0.5 * dsst_0
        X_mid["h"] = X["h"] + 0.5 * dwwv_0
        X_mid["t"] = X["t"] + 0.5 * dt

        # Step 3: 在中点重新计算导数
        dsst_mid = f.ddt_sst.eval(X_mid, **eval_args) * dt
        dwwv_mid = f.ddt_wwv.eval(X_mid, **eval_args) * dt
        return dsst_mid, dwwv_mid


class TrapezoidalIntegrator(Intergrator):
    """梯形法（时间可逆、隐式二阶）"""
    def __init__(self, max_iter=5, tol=1e-5):
        self.max_iter = max_iter
        self.tol = tol

    def step(self, f: Dynamics, X: Dict[str, np.ndarray], dt, **eval_args):
        # Step 1: 基于当前状态计算初始导数
        dsst_0 = f.ddt_sst.eval(X, **eval_args) * dt
        dwwv_0 = f.ddt_wwv.eval(X, **eval_args) * dt

        # Step 2: 预测初值（欧拉预测）
        X_1 = deepcopy(X)
        X_1["x"] = X["x"] + dsst_0
        X_1["h"] = X["h"] + dwwv_0
        X_1["t"] = X["t"] + dt

        # Step 3: 固定点迭代求解隐式部分
        for _ in range(self.max_iter):
            dsst_1 = f.ddt_sst.eval(X_1, **eval_args) * dt
            dwwv_1 = f.ddt_wwv.eval(X_1, **eval_args) * dt
            x_next = X["x"] + 0.5 * (dsst_0 + dsst_1)
            h_next = X["h"] + 0.5 * (dwwv_0 + dwwv_1)
            if np.allclose(x_next, X_1["x"], atol=self.tol) and np.allclose(h_next, X_1["h"], atol=self.tol):
                break
            else:
                X_1["x"], X_1["h"] = x_next, h_next

        # Step 4: 计算最终导数（近似平均）
        dsst_final = 0.5 * (dsst_0 + f.ddt_sst.eval(X_1, **eval_args) * dt)
        dwwv_final = 0.5 * (dwwv_0 + f.ddt_wwv.eval(X_1, **eval_args) * dt)
        return dsst_final, dwwv_final


class RK4Integrator(Intergrator):
    """四阶显式龙格-库塔法（高精度通用方案）"""
    def step(self, f: Dynamics, X: Dict[str, np.ndarray], dt, **eval_args):
        # k1
        dsst1 = f.ddt_sst.eval(X, **eval_args) * dt
        dwwv1 = f.ddt_wwv.eval(X, **eval_args) * dt

        # k2
        s2 = deepcopy(X)
        s2["x"] = X["x"] + 0.5 * dsst1
        s2["h"] = X["h"] + 0.5 * dwwv1
        s2["t"] = X["t"] + 0.5 * dt
        dsst2 = f.ddt_sst.eval(s2, **eval_args) * dt
        dwwv2 = f.ddt_wwv.eval(s2, **eval_args) * dt

        # k3
        s3 = deepcopy(X)
        s3["x"] = X["x"] + 0.5 * dsst2
        s3["h"] = X["h"] + 0.5 * dwwv2
        s3["t"] = X["t"] + 0.5 * dt
        dsst3 = f.ddt_sst.eval(s3, **eval_args) * dt
        dwwv3 = f.ddt_wwv.eval(s3, **eval_args) * dt

        # k4
        s4 = deepcopy(X)
        s4["x"] = X["x"] + dsst3
        s4["h"] = X["h"] + dwwv3
        s4["t"] = X["t"] + dt
        dsst4 = f.ddt_sst.eval(s4, **eval_args) * dt
        dwwv4 = f.ddt_wwv.eval(s4, **eval_args) * dt

        # Combine
        dsst = (dsst1 + 2 * dsst2 + 2 * dsst3 + dsst4) / 6.0
        dwwv = (dwwv1 + 2 * dwwv2 + 2 * dwwv3 + dwwv4) / 6.0
        return dsst, dwwv
