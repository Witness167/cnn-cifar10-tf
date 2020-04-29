import tensorflow as tf
import os

#使用GPU进行训练
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpuConfig = tf.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth = True

import _pickle as pickle
import numpy as np

def unpickle(filename):
    with open(filename, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
        return d

def onehot(labels):
    '''one-hot 编码'''
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels

#读取数据集 原始数据为二进制文件
# 训练数据集
data1 = unpickle('cifar-10-batches-py/data_batch_1')
data2 = unpickle('cifar-10-batches-py/data_batch_2')
data3 = unpickle('cifar-10-batches-py/data_batch_3')
data4 = unpickle('cifar-10-batches-py/data_batch_4')
data5 = unpickle('cifar-10-batches-py/data_batch_5')
X_train = np.concatenate((data1['data'], data2['data'], data3['data'], data4['data'], data5['data']), axis=0)
y_train = np.concatenate((data1['labels'], data2['labels'], data3['labels'], data4['labels'], data5['labels']), axis=0)
y_train = onehot(y_train)
# 测试数据集
test = unpickle('cifar-10-batches-py/test_batch')
X_test = test['data'][:5000, :]
y_test = onehot(test['labels'])[:5000, :]
print('Training dataset shape:', X_train.shape)
print('Training labels shape:', y_train.shape)
print('Testing dataset shape:', X_test.shape)
print('Testing labels shape:', y_test.shape)

#TODO 6
"""
可以调整超参数来寻找最合适的参数值，从而提高模型的效果
例如可以增大训练迭代次数或者更改学习率设置，开始的时候使用较大学习率，在迭代一定次数后缩小学习率，可以达到加快模型训练的效果
"""
learning_rate = 1e-3
training_iters = 200
batch_size = 50
display_step = 5
n_features = 3072  # 32*32*3
n_classes = 10
n_fc1 = 384
n_fc2 = 192


import time
import matplotlib.pyplot as plt

# 构建模型
x = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.float32, [None, n_classes])

W_conv = {
    'conv1': tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.0001)),
    'conv2': tf.Variable(tf.truncated_normal([5, 5, 32, 64],stddev=0.01)),
    'fc1': tf.Variable(tf.truncated_normal([8*8*64, n_fc1], stddev=0.1)),
    'fc2': tf.Variable(tf.truncated_normal([n_fc1, n_fc2], stddev=0.1)),
    'fc3': tf.Variable(tf.truncated_normal([n_fc2, n_classes], stddev=0.1))
}
b_conv = {
    'conv1': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[32])),
    'conv2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[64])),
    'fc1': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc1])),
    'fc2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc2])),
    'fc3': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_classes]))
}

#TODO 1
"""
数据增强：
    1 对图像进行随机翻转
    2 随机变换图像亮度
    3 随机变换图像对比度。。。
数据预处理：
    1 均值滤波
    2 中值滤波。。。
"""
x_image = tf.reshape(x, [-1, 32, 32, 3])

#TODO 2
"""
增加模型宽度和深度
调整卷积核的数量和卷积层的数量，寻找一个合适的数值。数量的增加可以增加模型的拟合能力，但是过大又可能造成过拟合的效果
"""
# 卷积层 1
conv1 = tf.nn.conv2d(x_image, W_conv['conv1'], strides=[1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, b_conv['conv1'])

#TODO 3
"""
不同的激活函数可以达到不同的非线性变换效果
"""
conv1 = tf.nn.relu(conv1)

#TODO 4
"""
池化层作用：
    1 保留主要特征，减少下一次的参数数量，防止过拟合
    2 保持某种不变形，例如平移不变形，常用的有mean-pooling max-pooling
"""
# 池化层 1
pool1 = tf.nn.avg_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

#TODO 5
"""
dropout以及不同的归一化
dropout通过在网络前向传播时让某个神经元以一定的概率p停止活动，使得模型减少对于部分特征的依赖。可以防止过拟合的效果
LRN:局部响应归一化层，增大反馈较大的神经元的反馈结果，提高模型泛化能力
可以使用批归一化BN来代替
"""
# LRN层，Local Response Normalization
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
# 卷积层 2
conv2 = tf.nn.conv2d(norm1, W_conv['conv2'], strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, b_conv['conv2'])
conv2 = tf.nn.relu(conv2)
# LRN层，Local Response Normalization
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
# 池化层 2
pool2 = tf.nn.avg_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
reshape = tf.reshape(pool2, [-1, 8*8*64])

fc1 = tf.add(tf.matmul(reshape, W_conv['fc1']), b_conv['fc1'])
fc1 = tf.nn.relu(fc1)
# 全连接层 2
fc2 = tf.add(tf.matmul(fc1, W_conv['fc2']), b_conv['fc2'])
fc2 = tf.nn.relu(fc2)
# 全连接层 3, 即分类层
fc3 = tf.nn.softmax(tf.add(tf.matmul(fc2, W_conv['fc3']), b_conv['fc3']))

# 定义损失
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
# 评估模型
correct_pred = tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session(config=gpuConfig) as sess:
    sess.run(init)
    c = []
    total_batch = int(X_train.shape[0] / batch_size)
#    for i in range(training_iters):
    start_time = time.time()
    for i in range(200):
        for batch in range(total_batch):
            batch_x = X_train[batch*batch_size : (batch+1)*batch_size, :]
            batch_y = y_train[batch*batch_size : (batch+1)*batch_size, :]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        print(acc)
        c.append(acc)
        end_time = time.time()
        print('time: ', (end_time - start_time))
        start_time = end_time
        print("---------------{} onpech is finished-------------------".format(str(i)))
    print("Optimization Finished!")

    # Test
    test_acc = sess.run(accuracy, feed_dict={x: X_test, y: y_test})
    print("Testing Accuracy:", test_acc)
    plt.plot(c)
    plt.xlabel('Iter')
    plt.ylabel('Cost')
    plt.title('lr=%f, ti=%d, bs=%d, acc=%f' % (learning_rate, training_iters, batch_size, test_acc))
    plt.tight_layout()
    plt.savefig('cnn-tf-cifar10-%s.png' % test_acc, dpi=200)