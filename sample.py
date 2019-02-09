#バイトコードの無効化
import sys
sys.dont_write_bytecode = True

import numpy as xp

import tensor_link.function as F
import tensor_link.learning as L
from tensor_link import Tensor,Model

class Network(Model):
    def __init__(self):
        super().__init__()
        pass
    def __call__(self, a,b,c,x):
        #return ax^2+bx+c
        h = F.product(a,x)
        h = F.add(h,b)
        h = F.product(h,x)
        h = F.add(h,c)
        
        return h

a = xp.array([[1,2],[3,4]])
a = Tensor(a)
b = xp.array([[1,2],[5,6],[3,4]])
b = Tensor(b)
test = F.affine(a,b)
test.backward()

print(a.grad)
print(b.grad)
print(test.data)