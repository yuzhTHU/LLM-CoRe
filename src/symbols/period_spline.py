from copy import deepcopy
from nd2py import Symbol, NetType
from scipy.interpolate import CubicSpline
from nd2py.core.calc.numpy_calc import NumpyCalc


class PeriodicSpline(Symbol):
    n_operands = 1

    def __init__(self, *operands, x, y, nettype: NetType = "scalar"):
        """
        - x: (L,) 节点位置列表, 必须是单调递增的
        - y: (L, N or 1) 节点值列表, 第一个值必须等于最后一个值 (或者长度比 x 少 1, 此时会自动补齐最后一个值)
        """
        super().__init__(*operands, nettype=nettype)
        if len(y) + 1 == len(x):
            y = [*y, y[0]]
        elif y[0] != y[-1]:
            raise ValueError("For periodic spline, the first and last y values must be the same.")
        self.spline = CubicSpline(x, y, bc_type="periodic")

    def copy(self):
        other = PeriodicSpline([0, 1], [0, 0], *self.operands, nettype=self.nettype)
        other.spline = deepcopy(self.spline)
        return other

    @staticmethod
    def visit_PeriodicSpline(node, *args, **kwargs):
        x = node.operands[0]
        result = yield (x, args, kwargs)
        return node.spline(result.squeeze(-1))

    NumpyCalc.visit_PeriodicSpline = visit_PeriodicSpline

    @classmethod
    def nettype_range(cls):
        return {"scalar", "node", "edge"}

    @classmethod
    def map_nettype(cls, operand_types):
        """Since nettype of this Symbol cannot be determined based on its operands."""
        return None
