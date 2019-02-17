#coding=utf-8
import tensorflow as tf
import numpy as np
print("tf %s" %tf.__version__)
print("numpy %s" %np.__version__)
#关于交叉熵的值的比较

v1 = tf.constant([1.0,0.0])
v2 = tf.constant([0.6,0.4])
v3 = tf.constant([0.8,0.2])
#求交叉熵的公式p1*log(p2)

res1 = -tf.reduce_sum(v1*tf.log(tf.clip_by_value(v2,1e-12,1.0)))
res2 = -tf.reduce_sum(v1*tf.log(tf.clip_by_value(v3,1e-12,1.0)))
sess = tf.Session()

print(sess.run(tf.log(v2)))
print(sess.run(res1))
print(sess.run(res2))
sess.close()
