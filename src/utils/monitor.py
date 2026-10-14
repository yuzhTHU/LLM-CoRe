import logging
from rich.text import Text
from rich.progress import ProgressColumn, Task, RenderableType

_logger = logging.getLogger(__name__)


class RichLogHandler(logging.Handler):
    def __init__(self, console, formatter):
        super().__init__()
        self.console = console
        self.formatter = formatter

    def emit(self, record):
        try:
            msg = self.formatter.format(record)
            self.console.print(Text.from_ansi(msg))
        except Exception:
            self.handleError(record)


class MergingBarColumn(ProgressColumn):
    """双向汇合进度条：左边成功（绿），右边失败（红），中间灰色未完成"""

    def __init__(self, width: int = 40):
        super().__init__()
        self.width = width

    def render(self, task: Task) -> RenderableType:
        success = task.fields.get("success", 0)
        failure = task.fields.get("failure", 0)
        total = task.total or (success + failure + 1)

        bar = Text()
        half = self.width // 2

        left_count = round(half * success / total * 2)
        right_count = round(half * failure / total * 2)

        # 剩余中间区域宽度
        middle_count = max(0, self.width - left_count - right_count)

        # 成功进度（左 → 中）
        if left_count > 0:
            bar.append("━" * (left_count - 1), style="green")
            bar.append("╸", style="green")

        # 中间灰色未完成区域
        if middle_count > 0:
            bar.append("━" * (middle_count - 1), style="dim")
            bar.append("╸", style="dim")

        # 失败进度（右 → 中）
        if right_count > 0:
            bar.append("━" * (right_count), style="red")

        return bar

