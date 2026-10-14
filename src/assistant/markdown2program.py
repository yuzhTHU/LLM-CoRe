import ast
import logging
from nd2py.search.llmsr import render_markdown
from abc import ABC, abstractmethod
from ..utils import log_exception
from .frozen_base import FrozenMixin

_logger = logging.getLogger(__name__)


class Markdown2Program(FrozenMixin, ABC):
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
    def __call__(self, markdown):
        """ 尝试从 markdown 中提取有效的 (语法正确的) 函数定义部分 """
        raise NotImplementedError


@Markdown2Program.register('default')
class Markdown2Program_v0(Markdown2Program):
    def __call__(self, markdown):
        lines = markdown.split("\n")
        ## Find the first line of the function definition
        for start_idx, line in enumerate(lines):
            if line.startswith("def "):
                break
        else:
            _logger.trace(
                f"No function definition found in the program:"
                f"\n{render_markdown(markdown, theme='dracula')}"
            )
            return None
        lines = lines[start_idx:]
        ## Find the end of the function definition
        end_idx = len(lines)
        while end_idx > 1:
            try:
                ast.parse("\n".join(lines[:end_idx]))
                break
            except SyntaxError as e:
                if e.lineno is not None:
                    end_idx = e.lineno - 1  # e.lineno 是从 1 开始数的
                else:
                    _logger.trace(
                        f"Syntax error in the program since "
                        f"{log_exception(e)}\n{render_markdown(markdown, theme='dracula')}"
                    )
                    return None
        else:
            _logger.trace(
                f"No valid function definition found in the program:"
                f"\n{render_markdown(markdown, theme='dracula')}"
            )
            return None
        lines = lines[:end_idx]
        ## Ensure the function definition has a return statement
        program = "\n".join(lines)
        if "return " not in program:
            _logger.trace(
                f"No return statement found in the function definition:"
                f"\n{render_markdown(markdown, theme='dracula')}"
            )
            return None
        return program
