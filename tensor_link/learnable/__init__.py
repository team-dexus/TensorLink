import tensor_link
from tensor_link import Tensor
from tensor_link import xp
#LearnableはLearnableクラスを継承して作るといいと思う。
class Learnable:
    def __init__(self):
        pass
    def __call__(self):
        pass
    def get_all_params(self):
        params = []
        for obj in self.__dict__.values():
            if issubclass(obj.__class__, Tensor):
                if obj.train:
                    params.append(obj)
        return params
    def zero_grads(self):
        for obj in self.__dict__.values():
            if issubclass(obj.__class__, Tensor):
                obj.grad = xp.zeros_like(obj.data,dtype=tensor_link.config.DEFAULT_DTYPE)


#作ったLearnableは全てここでimport
from tensor_link.learnable.layer.affine import Affine
