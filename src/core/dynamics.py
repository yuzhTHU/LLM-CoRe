import nd2py as nd


class Dynamics:
    """ 演化动力学 """
    def __init__(self, ddt_sst: nd.Symbol, ddt_wwv: nd.Symbol):
        self.ddt_sst = ddt_sst
        self.ddt_wwv = ddt_wwv
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return (
            f"Dynamics(dx_dt={self.ddt_sst.to_str(number_format='.2f')}, "
            f"dh_dt={self.ddt_wwv.to_str(number_format='.2f')})"
        )
