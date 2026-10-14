import ast
import textwrap
from nd2py.search.llmsr import render_python


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
