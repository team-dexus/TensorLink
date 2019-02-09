from tensor_link import xp
from tensor_link import Tensor

class Normal:
    def __init__(self,std=0.2):
        self.std = std
    def __call__(self,shape):
        return Tensor(xp.random.normal(0, self.std, shape))
        