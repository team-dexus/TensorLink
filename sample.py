#バイトコードの無効化
import sys
sys.dont_write_bytecode = True

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

x = Tensor(5)
a = Tensor(2)
b = Tensor(3)
c = Tensor(5)

test = Network()
loss = test(a,b,c,x)
loss.backward()

print(loss.data)
print(a.grad)
print(b.grad)
print(c.grad)
print(x.grad)