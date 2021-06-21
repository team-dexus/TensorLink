from tensor_link import Tensor
from tensor_link.function import Function
from tensor_link import xp


class _affine(Function):        
    def __init__(self):
        super().__init__()

    def __call__(self,a,b):
        new_tensor = Tensor(xp.dot(a.data,b.data.T).T)
        self.graph = [a,b]
        new_tensor.graph=self
        return new_tensor

    def backward(self,grad):
        self.graph[0].grad += xp.dot(self.graph[1].data.T,grad).T
        self.graph[1].grad += xp.dot(grad,self.graph[0].data)  
        self.graph[0].backward(self.graph[0].grad)
        self.graph[1].backward(self.graph[1].grad)

def affine(a,b):
    return _affine()(a,b)

'''
a b x  y   ax+by   cx+dy   g1 g2 ←これらが流れてくる a:+=g1x+g3x1 b:g1y+g3y1 x:+=ag1+cg2 x1:+=bg1+dg2
c d x1 y1  ax1+by1 cx1+dy1 g3 g4 次のノードに流す値→ c:+=g2x+g4x1 d:g2y+g4y1 y:+=ag3+cg4 y1:+=bg3+dg4
'''