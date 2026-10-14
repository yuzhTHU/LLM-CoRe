import nd2py as nd


class Triplet:
    """ 拆分的演化动力学 """
    def __init__(self, coeff: nd.Symbol, f_sst: nd.Symbol, f_wwv: nd.Symbol):
        self.coeff = coeff
        self.f_sst = f_sst
        self.f_wwv = f_wwv
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"Triplet(coeff={self.coeff.to_str(number_format='.2f')}, "
            f"f_sst={self.f_sst.to_str(number_format='.2f')}, "
            f"f_wwv={self.f_wwv.to_str(number_format='.2f')})"
        )
