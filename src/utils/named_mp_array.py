import multiprocessing as mp

class NamedMpArray:
    def __init__(self, names, typecode='i'):
        # 注意：这里假设 names 在初始化后不会改变
        self.name_to_idx = {name: i for i, name in enumerate(names)}
        self.array = mp.Array(typecode, len(names))
    
    def __getitem__(self, name):
        return self.array[self.name_to_idx[name]]
    
    def __setitem__(self, name, value):
        self.array[self.name_to_idx[name]] = value
    
    def to_dict(self):
        with self.array.get_lock():
            return {name: self.array[idx] for name, idx in self.name_to_idx.items()}

    def inc(self, name, delta=1):
        idx = self.name_to_idx[name]
        with self.array.get_lock(): # 锁住整个读-改-写过程
            self.array[idx] += delta

    def dec(self, name, delta=1):
        idx = self.name_to_idx[name]
        with self.array.get_lock(): # 锁住整个读-改-写过程
            self.array[idx] -= delta

    def get_lock(self):
        return self.array.get_lock()
