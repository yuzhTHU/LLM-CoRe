
class GenWrapper:
    """Wrap an iterator so that StopIteration.value can be captured."""
    def __init__(self, gen):
        self._gen = gen
        self.return_value = None  # 用来存 return value

    def __iter__(self):
        self.return_value = yield from self._gen
