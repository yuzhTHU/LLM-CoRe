import numpy as np
from nd2py import Symbol, NetType
from nd2py.core.calc.numpy_calc import NumpyCalc, yield_nothing


class WhiteNoise(Symbol):
    n_operands = 0

    def __init__(self, shape, nettype: NetType = "scalar"):
        super().__init__(nettype=nettype)
        self.shape = shape
        self.current = np.random.normal(size=shape)

    def copy(self):
        other = WhiteNoise(self.shape, nettype=self.nettype)
        return other

    @staticmethod
    def visit_WhiteNoise(node, *args, **kwargs):
        # 如果你想兼容框架中的 yield-nothing pattern，也可以加上下面这行：
        yield from yield_nothing()
        return np.random.normal(size=node.shape)

    NumpyCalc.visit_WhiteNoise = visit_WhiteNoise

    @classmethod
    def create_from_residuals(cls, residuals, shape):
        """Create a WhiteNoise instance from residuals."""
        return cls(shape, nettype="scalar")
