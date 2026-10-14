import numpy as np
import pandas as pd
from nd2py import Symbol, NetType
from nd2py.core.calc.numpy_calc import NumpyCalc, yield_nothing


class RedNoise(Symbol):
    n_operands = 0

    def __init__(self, a, shape, nettype: NetType = "scalar", random_state=None):
        super().__init__(nettype=nettype)
        self.a = a
        self.b = np.sqrt(1 - a**2)
        self.shape = shape
        self._rng = np.random.default_rng(random_state)
        self.current = self._rng.normal(size=shape)

    def copy(self):
        other = RedNoise(self.a, self.shape, nettype=self.nettype)
        other.current = self.current.copy()
        return other

    @staticmethod
    def visit_RedNoise(node, *args, **kwargs):
        yield from yield_nothing()
        size = [kwargs['length'], *node.shape[1:]] if 'length' in kwargs else node.shape
        source = node._rng.normal(size=size)
        node.current = node.a * node.current + node.b * source
        return node.current

    NumpyCalc.visit_RedNoise = visit_RedNoise

    @classmethod
    def nettype_range(cls):
        return {"scalar", "node", "edge"}

    @staticmethod
    def create_from_residuals(
        residuals: pd.DataFrame,
        n=1,
        max_lag=2,
        shape=[1],
        nettype="scalar",
        random_state=42,
    ) -> "RedNoise":
        """Create a RedNoise instance from residuals."""
        x = residuals - residuals.mean()
        var = np.var(x, axis=0, keepdims=True)
        a1_dict = {}
        for lag in range(1, max_lag + 1):
            corr = (x * x.shift(-lag)).mean()
            a1_dict[lag] = (corr / var) ** (1 / lag)
        a1 = pd.DataFrame(a1_dict).T.mean()
        a1 = np.sign(a1) * np.abs(a1) ** (1 / n)  # Adjust for the number of steps
        return RedNoise(a=a1, shape=shape, nettype=nettype, random_state=random_state)
