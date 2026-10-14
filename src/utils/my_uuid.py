import uuid
import numpy as np

def my_uuid(rng: np.random.Generator):
    """
    使用 numpy.random.Generator 生成可控的 UUID4。
    相同 RNG 状态生成相同 UUID。
    """
    # 生成 128 位随机数
    # np.random.Generator 不直接生成 128 位整数，所以生成两个 64 位再拼接
    high = rng.integers(0, 2**64, dtype=np.uint64)
    low = rng.integers(0, 2**64, dtype=np.uint64)
    rand_bits = (int(high) << 64) | int(low)
    
    # 设置版本位 (4) - bits 12-15
    rand_bits &= ~(0xf << 76)
    rand_bits |= (4 << 76)
    
    # 设置变体位 (10xx) - bits 64-65
    rand_bits &= ~(0b11 << 62)
    rand_bits |= (0b10 << 62)
    
    return str(uuid.UUID(int=rand_bits))
