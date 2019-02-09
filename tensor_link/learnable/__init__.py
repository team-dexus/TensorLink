#LearnableはLearnableクラスを継承して作るといいと思う。
class Learnable:
    def __init__(self):
        pass
    def __call__(self):
        pass


#作ったLearnableは全てここでimport
from tensor_link.learnable.layer.affine import Affine
