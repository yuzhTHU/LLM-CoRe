import logging
from abc import ABCMeta

_logger = logging.getLogger(__name__)


class FrozenABCMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        # object.__setattr__(instance, '_frozen', True)
        instance._frozen = True
        return instance


class FrozenMixin(metaclass=FrozenABCMeta):
    def __setattr__(self, key, value):
        # 如果 _frozen 标记存在且为 True，则禁止修改
        if getattr(self, '_frozen', False):
            msg = f"Instance of {type(self).__name__} is frozen. Cannot modify '{key}'."
            # raise AttributeError(msg)
            _logger.error(msg)
        super().__setattr__(key, value)

    def __delattr__(self, key):
        # 如果 _frozen 标记存在且为 True，则禁止删除
        if getattr(self, '_frozen', False):
            msg = f"Instance of {type(self).__name__} is frozen. Cannot delete '{key}'."
            # raise AttributeError(msg)
            _logger.error(msg)
        super().__delattr__(key)
