# Tensor Linkの仕様
## 1.記法
	import TensorLink.Learning as L
	import TensorLink.Function as F
	import train_data
	import teacher_data

	class CNN:
		def __init__(self):
			self.c1 = L.conv2D(3,128,3,1,1)
			self.c2 = L.conv2D(128,1,3,1,1)
			self.l1 = L.Dense(None,2)
		def __call__(self, x):
			h = self.c1(x)
			h = self.c2(h)
			h = self.l1(h)

			return h
	opt = Adam(0.001,0.99,0.99)
	network = CNN()
	while True:
		result = CNN(train_data)
		loss = F.mse(result,teacher_data)
		opt.update(network,loss)#networkを、lossの誤差逆伝播をもとに更新、の意。
		print(loss.data)

## 2.クラス概要
Tensor → 計算グラフを保持した変数（ベクトル、二次元配列なども値として入る。）  
Function → 学習不可な関数  
Learning → 学習可能な関数。TensorクラスとFunctionクラスの組み合わせで作られる。  

## 3.Tensorクラス
#### 変数
graph → 計算グラフを何らかの形で保持  
data → 生の値  
grad → 勾配を格納する  
#### 関数
backward() → 誤差逆伝播を行う。  
## 4.Functionクラス
（考え中）
## 5.Learningクラス
（考え中）
