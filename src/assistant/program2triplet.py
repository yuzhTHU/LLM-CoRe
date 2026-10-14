import ast
import logging
import textwrap
import nd2py as nd
from nd2py.search.llmsr import render_python
from abc import ABC, abstractmethod
from ..utils import log_exception
from ..core import Triplet
from .frozen_base import FrozenMixin

_logger = logging.getLogger(__name__)


class Program2Triplet(FrozenMixin, ABC):
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
    def create(cls, assistant):
        sub_class = cls.SUBCLASS_REGISTRY.get("default")
        return sub_class(assistant)

    @abstractmethod
    def __call__(self, program):
        """ 将 program 代码转换成对应的 triplet 三元组, 需要大致保证可逆 """
        raise NotImplementedError


@Program2Triplet.register('default')
class Program2Triplet_v0(Program2Triplet):
    def __call__(self, program):
        ## Check Validaty
        try:
            tree = ast.parse(program)
        except SyntaxError as e:
            _logger.trace(
                f"Syntax error in the program since:"
                f"{log_exception(e)}\n{render_python(program, theme='dracula')}"
            )
            return None
        ## Ban dangerous operation
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                _logger.trace(
                    f"Dangerous operation (import) found in the program:"
                    f"\n{render_python(program, theme='dracula')}"
                )
                return None
        ## Fix function signature
        try:
            new_program, function_name = self.combine_signature_and_body(program, self.assistant.signature)
        except Exception as e:
            _logger.trace(
                f"Error combining function signature and body since: "
                f"{log_exception(e)}\n{render_python(program, theme='dracula')}"
            ) 
            return None
        ## Execute program to get function
        try:
            safe_globals = self.assistant.namespace.copy()
            safe_globals["__builtins__"] = {}
            exec(new_program, safe_globals, (playground := {}))
            function = playground[function_name]
            with nd.no_nettype_check(): # , nd.no_set_fitable():
                coeff, f_sst, f_wwv = function(**self.assistant.signature_args)
        except Exception as e:
            _logger.trace(
                f"Error parsing program below since: "
                f"{log_exception(e)}\n{render_python(new_program, theme='dracula')}"
            )
            return None
        ## Fix non-Symbols
        if not isinstance(coeff, nd.Symbol):
            coeff = nd.Number(coeff, fitable=True)
        if not isinstance(f_sst, nd.Symbol):
            f_sst = nd.Number(f_sst, fitable=True)
        if not isinstance(f_wwv, nd.Symbol):
            f_wwv = nd.Number(f_wwv, fitable=True)
        ## Fix incorrect nettypes
        coeff = coeff.fix_nettype(nettype=self.assistant.variables['t'].nettype, direction="bottom-up")
        f_sst = f_sst.fix_nettype(nettype=self.assistant.variables['x'].nettype, direction="bottom-up")
        f_wwv = f_wwv.fix_nettype(nettype=self.assistant.variables['h'].nettype, direction="bottom-up")
        ## Return parsed (coeff, f_sst, f_wwv)
        return Triplet(coeff=coeff, f_sst=f_sst, f_wwv=f_wwv)

    @staticmethod
    def combine_signature_and_body(program: str, signature: str) -> str:
        """ 将 program 和函数体与 signature 组合成一个函数，返回新的函数及其名称 """
        ## 清理 signature
        signature = signature.strip()
        if not signature.endswith(':'):
            signature = signature + ':'
        ## 解析 program
        try:
            tree = ast.parse(program)
        except SyntaxError as e:
            raise SyntaxError(f"Failed to parse program: {e.msg}") from e
        ## 查找函数定义节点 (ast.FunctionDef)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                break
        else:
            raise ValueError("Failed to find function defination in program")
        ## 提取函数体
        # 获得无缩进的函数体代码
        body_wo_indent = ast.unparse(node.body)
        # 处理空函数体, 例如 def f(): pass
        if not body_wo_indent.strip():
            body_wo_indent = "pass"
        # 添加缩进
        body_with_indent = textwrap.indent(body_wo_indent, "    ")
        ## 拼接为完整函数并返回
        new_program = f"{signature}\n{body_with_indent}"
        return new_program, signature.strip().removeprefix('def ').split('(', 1)[0]
