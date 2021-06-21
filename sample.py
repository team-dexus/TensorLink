#バイトコードの無効化
import sys
sys.dont_write_bytecode = True

import numpy as xp

import tensor_link
import tensor_link.function as F
import tensor_link.learnable as L
from tensor_link import Tensor,Model

class Network(Model):
    def __init__(self):
        super().__init__()
        self.a1 = L.Affine(3,3)
    def __call__(self,x):
        h = self.a1(x)
        h = F.product(h,h)
        
        return h


opt = tensor_link.optimizer.SGD(0.001)
net = Network()
opt.assign(net)
x = Tensor(xp.array([[1,2,5],[3,4,2]]))

for i in range(20):
    loss = net(x)

    net.zero_grads()
    loss.backward()
    opt.update()

    print(loss.data.sum())
    print(net.a1.weight.data)
    print(net.a1.bias.data)

'''
NetworkはxをAffineレイヤーに通して、その後要素ごとに二乗している。
学習部分では、出力の全ての要素を最小化しようとしている
するとその合計も当然、最小値である0に近づいていく
'''