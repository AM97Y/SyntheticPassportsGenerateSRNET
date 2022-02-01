import tensorflow as tf
a = tf.Variable(5)
b = tf.Variable(6)
c = tf.Variable(7)
d = (a + b) * c

for i in tf.get_default_graph().get_operations():
    print(i.name)