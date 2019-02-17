#coding=utf-8
import tensorflow as tf
import numpy as np

print("tf %s" %tf.__version__)
print("numpy %s" %np.__version__)

BATCH_SIZE = 8
#seed = 3
seed = 23455
#准备数据
rng = np.random.RandomState(seed)

X = rng.rand(32,2)
Y = [[int(x0+x1<1)]for (x0,x1) in X]

print("X %s \n" %X)
print("Y %s \n" %Y)
#搭建神经网络
x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))

w1 = tf.Variable(tf.random_normal(shape=(2,3),stddev=1,seed=2))
w2 = tf.Variable(tf.random_normal(shape=(3,1),stddev=1,seed=2))

a1 = tf.matmul(x,w1)
y = tf.matmul(a1,w2)

loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

STEP = 3000
#进行计算
with tf.Session() as sess:
    var_init = tf.global_variables_initializer()
    sess.run(var_init)
    print(sess.run(w1))
    print(sess.run(w2))
    for i in range(STEP):
        start = (i*BATCH_SIZE)%32
        end = start + BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%500==0:#得出结果
            total_loss = sess.run(loss,feed_dict={x:X,y_:Y})
            print("After %d training step(s),loss on all data is %g" %(i,total_loss))
    print(sess.run(w1))
    print(sess.run(w2))
#得出结果