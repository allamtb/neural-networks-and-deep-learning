# coding=utf-8
import pickle
import time

import matplotlib
import mnist_loader
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors


def main():
    f = open('myNeural.txt', 'r')
    net = pickle.load(f)
    # 测试神经元网络的正确性。
    # 1、取出一组测试数据
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    knn = init_KNN(training_data)
    # 2、图形化展示测试数据
    for i in test_data:
        testdata = i[0]
        # 3、使用机器学习算法来验证测试数据
        # a、使用神经元网络验证

        result = np.argmax(net.feedforward(testdata))
        print result
        print i[1]

        # b、使用KNN来验证
        predict = knn.predict(np.reshape(testdata,(1,-1)))
        print predict
        # c、使用聚类来验证

        showFigure(testdata, result, i[1])


def init_KNN(training_data):
    trains = [np.reshape(x, (1, -1))[0] for (x, y) in training_data]
    rightdata = [np.argmax(np.reshape(y, (1, -1))[0]) for (x, y) in training_data]
    knn = neighbors.KNeighborsClassifier()
    knn.fit(trains, rightdata)
    return knn


def showFigure(testdata,result,expectValue):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    testdata = np.reshape(testdata, (-1, 28))
    ax.matshow(testdata, cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.savefig(str(result)+'-'+str(expectValue))
    plt.show()



main()
