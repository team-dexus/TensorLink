from tensor_link import Tensor
from tensor_link.learnable import Learnable
from tensor_link.util import initializer 
import tensor_link.function as F

class Affine(Learnable):
    def __init__(self,in_size,out_size,w_init=None,b_init=None):
        super().__init__()
        if w_init is None:
            w_init = initializer.Normal()
        if b_init is None:
            b_init = initializer.Normal()
        self.weight = w_init(shape = (out_size,in_size))
        self.bias = b_init(shape = (out_size,))
        self.weight.train = True
        self.bias.train = True
    def __call__(self,x):
        return F.add(F.affine(self.weight,x),self.bias)
