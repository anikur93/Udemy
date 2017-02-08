import numpy as np
import tensorflow as tf

A = tf.placeholder(tf.float32, shape = (5,5), name ='A')

v = tf.placeholder(tf.float32)

w = tf.matmul(A,v)

with tf.Session() as session:
    output = session.run(w, feed_dict={A:np.random.randn(5,5),v:np.random.randn(5,1)})
    print(output,type(output))
    

    
x = tf.Variable(tf.random_normal((2,2)))
t = tf.Variable(0)

init = tf.initialize_all_variables()

with tf.Session() as session:
    out = session.run(init)
    print(out)
    print(x.eval())
    print(t.eval())
    
u = tf.Variable(20.0)
cost = u*u + u + 1
train_op = tf.train.GradientDescentOptimizer(0.3).minimize(cost)

init = tf.initialize_all_variables()
with tf.Session() as session:
    session.run(init)
    for i in range(20):
        session.run(train_op)
        print(cost.eval(), u.eval())