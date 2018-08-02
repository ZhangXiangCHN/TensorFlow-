# mnist demo

mnist 数字识别的简单实现（全连接DNN实现），主要注意的是使用了滑动平均模型来增加了模型的鲁棒性。

主要包括3个文件：

mnist_inference.py   深度神经网络模型的前向传播过程，这里简单的使用了一个隐藏层

mnist_train.py   模型的训练过程

mnist_eval.py   模型的评估过程，模型一直获取最新的模型进行评估
