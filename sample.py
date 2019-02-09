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
        self.a1 = L.Affine(2,2)
    def __call__(self,x):
        h = self.a1(x)
        
        return h


net = Network()
x = Tensor(xp.array([[1,2],[2,3]]))
test = net(x)
test.backward()
f_x = test.data.sum()

print(f_x)
print(net.a1.weight.grad)

#数値微分（検証用）
net.a1.weight.data[1][0] += 0.001
test = net(x)
print((test.data.sum() - f_x) / 0.001)

'''
出力例
0.137 #全結合層通したのち、全て足し合わせた数値
[[3. 5.]#weightの微分の値。
 [3. 5.]]
5.0048828125 #weightのなかのどれかで数値微分した結果。上の誤差逆伝播とこれがほぼ同じなら成功。違ったら何か間違ってる。
'''
