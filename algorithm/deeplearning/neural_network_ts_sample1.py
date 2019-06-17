import tensorflow as tf
import numpy as np
from numpy.random import RandomState
 
if __name__ == "__main__":
    #定义神经网络的参数
    w = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
    b = tf.Variable(tf.random_normal([1],stddev=1,seed=1))
    #定义输入和输出
    x = tf.placeholder(tf.float32,shape=(None,2),name="x-input")
    y_ = tf.placeholder(tf.float32,shape=(None,1),name="y-input")
    #定义神经网络的前向传播过程
    y = tf.nn.sigmoid(tf.matmul(x,w) + b)
    #基于min和max对张量t进行截断操作，为了应对梯度爆发或者梯度消失的情况
    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y_) * tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
    # 使用Adadelta算法作为优化函数，来保证预测值与实际值之间交叉熵最小
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    #通过随机函数生成一个模拟数据集
    rdm = RandomState(1)
    # 定义数据集的大小
    dataset_size = 100
    # 模拟输入是一个二维数组
    X = rdm.rand(dataset_size,2)
    #定义输出值，将x1+x2 < 1的输入数据定义为正样本
    Y = [[int(x1+x2 < 1)] for (x1,x2) in X]
    #创建会话运行TensorFlow程序
    with tf.Session() as sess:
        #初始化变量  tf.initialize_all_variables()
        init = tf.initialize_all_variables()
        sess.run(init)
        #设置神经网络的迭代次数
        steps = 500
        for i in range(steps):
            #通过选取样本训练神经网络并更新参数
           for (input_x,input_y) in zip(X,Y):
               input_x = np.reshape(input_x,(1,2))
               input_y = np.reshape(input_y,(1,1))
               sess.run(train_step,feed_dict={x:input_x,y_:input_y})
            #每迭代1000次输出一次日志信息
           if i % 100 == 0:
                # 计算所有数据的交叉熵
                total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
                # 输出交叉熵之和
                print("After %d training step(s),cross entropy on all data is %g"%(i,total_cross_entropy))
# =============================================================================
#         #预测输入X的类别
#         pred_Y = sess.run(y,feed_dict={x:X})
#         index = 1
#         for pred,real in zip(pred_Y,Y):
#             print(pred,real)
# =============================================================================
       
        #预测输入X的类别
        X1 = [[0.7, 0.2], [1.1, 2.0]]
        Y1 = [1, 0]
        pred_Y1 = sess.run(y,feed_dict={x:X1})
        index = 1
        for pred,real in zip(pred_Y1,Y1):
            print(pred,real)