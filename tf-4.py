#coding=utf-8
import tensorflow as tf
print("tf %s" %tf.__version__)
#使用指数衰减的学习率来训练参数
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_STEP = 1

global_step = tf.Variable(0.0,trainable=False)
#定义指数衰减的学习率
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,global_step,LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase=True)

w = tf.Variable(5.0,dtype=tf.float32)

#训练loss的两个步骤
loss = tf.square(w+1)
#这里还有global_step这个参数
train_step = tf.train.GradientDescentOptimizer(learning_rate).\
    minimize(loss,global_step=global_step)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        w2 = sess.run(w)
        learning_rate2 = sess.run(learning_rate)
        global_step2 = sess.run(global_step)
        loss2 = sess.run(loss)
        print("i:%s learning rate:%s global_step:%s w:%f loss:%f" %(i,learning_rate2,global_step2,w2,loss2))
        sess.run(train_step)