# coding=utf-8
import pickle
import time

import matplotlib
import mnist_loader
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.naive_bayes import BernoulliNB


def main():
    f = open('myNeural.txt', 'r')
    net = pickle.load(f)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # 处理为 scikit_learn所需的数据结构
    X_train = [np.reshape(x, (1, -1))[0] for (x, y) in training_data]
    y_train = [np.argmax(np.reshape(y, (1, -1))[0]) for (x, y) in training_data]

    # Fit estimators
    ESTIMATORS = {

        # KNN
        "K-nn": neighbors.KNeighborsClassifier().fit(X_train, y_train),
        # 朴素贝叶斯
        "native-bayes": BernoulliNB().fit(X_train, y_train)
        # 决策树
        # 聚类
    }

    for i in test_data:
        print '=================================='
        testdata = i[0]
        print '正确的结果为%d' % i[1]
        # 使用神经元网络验证
        result = np.argmax(net.feedforward(testdata))
        print '使用神经元网络分类的结果为', result
        for name, estimator in ESTIMATORS.items():
            print '使用%s进行分类，分类结果为%s' % (name, estimator.predict(np.reshape(testdata, (1, -1))))
            # 图形化展示测试数据
            #  showFigure(testdata, result, i[1])


def showFigure(testdata, result, expectValue):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    testdata = np.reshape(testdata, (-1, 28))
    ax.matshow(testdata, cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.savefig(str(result) + '-' + str(expectValue))
    plt.show()


## 问题1： 超大训练集如何分布式训练、增量训练？
main()
