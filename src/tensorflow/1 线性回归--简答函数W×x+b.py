# coding=utf-8
"""

   使用tensorflow来对方程 y= W*x +b 进行线性回归。通过输入一系列 (x,y)的向量，使用梯度下降方法来最小化预测值与y的差距来不断得到最优化的W/b向量。

   可以借鉴到几点：
   1. 对样本对归一化处理。
      a 可降低样本之间的差异。如果不归一化处理，仅能回归少量到样本。 归一化后，所能处理到样本能增加到99个。若不归一化，仅能处理5个。
      b 可使用 sklearn到preprocessing.scale来将样本统一为均值为0/方差为1 到正态分布曲线。
   2. 使用numpy来生成样本序列的几种方式。

      a. np.linspace(start,end,num)线性向量
      b. np.arange(100)
      c. np.random.normal 按给定方差/均值的正态分布生成随机数。
   3. 图形化结果


"""
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Make up some real data
x_data = np.linspace(1, 1000, 100) # 当第三个参数大于100时，回归出现紊乱。
noise = np.random.normal(0, 40, x_data.shape)
y_data = x_data  + noise*3

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

# Model parameters
W = tf.Variable([.1], tf.float32)
b = tf.Variable([-.1], tf.float32)

# Model input and output
x = tf.placeholder(tf.float32,shape=None)
linear_model = W * x + b
y = tf.placeholder(tf.float32)


# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = preprocessing.scale(x_data)
y_train = preprocessing.scale(y_data)
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong


for i in range(10000):
  sess.run([train], {x:x_train, y:y_train})
  if i % 50 == 0:
    # to visualize the result and improvement
    try:
        ax.lines.remove(lines[0])
    except Exception:
        pass
    prediction_value = sess.run(linear_model, feed_dict={x: x_data})
    # plot the prediction
    lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
    plt.pause(1)


