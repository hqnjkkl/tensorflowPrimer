#coding=utf-8
import tensorflow as tf
print("tf %s" %tf.__version__)
#关于梯度下降策略，如何影响参数值的改变
#w = tf.Variable(tf.constant(5,dtype=tf.float32))
w = tf.Variable(5.0)
#w = tf.constant(5,dtype=tf.float32)
#定义损失函数的公式
loss = tf.square(w+1)
learn_rate = 0.2
#定义梯度下降来找到loss的局部最小值，这个局部最小值也是全局最小值
train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

with tf.Session() as sess:
    init_opt = tf.global_variables_initializer()
    sess.run(init_opt)
    for i in range(0,40):
        sess.run(train_step)
        #得有这一步run，才能够显示它的值
        w2 = sess.run(w)
        print("i %s; w %s\n" %(i,w2))
