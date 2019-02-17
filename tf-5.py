#coding=utf-8
#模拟滑动平均的运行，滑动平均可以增强模型的泛化能力
import tensorflow as tf
#1.定义变量及滑动平均类
w1 = tf.Variable(0,dtype=tf.float32)

global_step = tf.Variable(0,dtype=tf.float32)

MOVING_AVERAGE_DECAY = 0.99

ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

ema_op = ema.apply(tf.trainable_variables())

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run([w1,ema.average(w1)]))
    #decay = min(0.99,(1+0)/(10+0))=0.1, 影子=0,w1=1
    #影子 = decay*旧影子+(1-decay)*w1 = 0.1*0+0.9*1 = 0.9
    #参数w1的赋值为1
    sess.run(tf.assign(w1,1))
    sess.run(ema_op)
    print(sess.run(global_step))
    print(sess.run([w1, ema.average(w1)]))

    # decay = min(0.99,(1+0)/(10+0))=0.1, 影子=0.9,w1=1
    # 影子 = decay*旧影子+(1-decay)*w1 = 0.1*0.9+0.9*1 = 0.9
    sess.run(ema_op)
    print(sess.run(global_step))
    print(sess.run([w1, ema.average(w1)]))

