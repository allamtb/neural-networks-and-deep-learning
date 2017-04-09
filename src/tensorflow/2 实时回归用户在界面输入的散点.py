# coding=utf-8

"""
   引入matplotlib的event在图形化界面中获取输入的点（scatter）,
   并实时的训练学习模型来进行线性回归，试图找出点之间的趋势。

   遗留问题：

      1.在这个场景下面，每次用户点击生成一个点（scatter）后，希望系统终止之前的训练，重新运行新的训练。
        目前不知道如何终止已经运行的tensorflow session。
      2. 如何获取 tensorflow session的运行状态？ 是正在运行中还是已经停止？

"""
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim(0, 50)
ax.set_xlim(0, 50)

mouseClickX = []
mouseClickY = []


cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()


def onclick(event):
    # 获取鼠标点击位置的坐标，将其存储。
    mouseClickX.append(event.xdata)
    mouseClickY.append(event.ydata)
    #重新展示所有的散点
    ax.cla()
    ax.scatter(mouseClickX, mouseClickY)
    # 固定坐标轴
    ax.set_ylim(0, 50)
    ax.set_xlim(0, 50)
    fig.canvas.draw()
    trainModel()


def trainModel():
    # Model parameters
    W = tf.Variable([.1000], tf.float32)
    b = tf.Variable([-.1000], tf.float32)
    # Model input and output
    x = tf.placeholder(tf.float32, shape=None)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)
    # loss
    loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    # training data
    x_train = preprocessing.scale(mouseClickX)
    y_train = preprocessing.scale(mouseClickY)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)  # reset values to wrong
    for i in range(500):
        sess.run([train], {x: x_train, y: y_train})
        if i % 50 == 0:
            # to visualize the result and improvement
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            print(x_train, y_train, i)
            prediction_value = sess.run(linear_model, feed_dict={x: mouseClickX})
            # plot the prediction
            lines = ax.plot(mouseClickX, prediction_value, 'r-', lw=5)
            plt.pause(1)
