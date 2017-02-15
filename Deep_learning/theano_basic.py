import theano.tensor as T

s = T.scalar('s')

v = T.vector('v')

m = T.matrix('m')

w = m.dot(v)

import theano

matrix_times_vector = theano.function(inputs = [m, v], outputs = w)

import numpy as np

v1 = np.array([5,6])

m1 = np.array([[1,2],[3,4]])

w1 = matrix_times_vector(m1,v1)

print(w1)

#shared variables for updates
x = theano.shared(20.0, 'x')

cost = x*x + x + 1

x_update = x - 0.3*T.grad(cost, x)

train = theano.function(inputs=[], outputs=cost, updates=[(x, x_update)])

for i in range(20):
    cost_val = train()
    print(cost_val,x.get_value())

