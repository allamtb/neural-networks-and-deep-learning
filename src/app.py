# coding=utf-8
__author__ = 'allamtb'

import network
import mnist_loader
import pickle
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 10])
net.SGD(training_data, 30, 1000, 3.0, test_data=test_data)
f = open('myNeural.txt', 'w')
pickle.dump(net,f)



# 学习其原理 ，  一步步的理解、注释。  ---  计划 11.18 看看书籍后，再来回顾。
# 构造自己的代码。 python 优美就优美在，代码的数学公式化。
# 可视化展示。输入、输出。
# 动态学习 -- online learn 以及 集群思考




