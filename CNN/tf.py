import tensorflow as tf

with tf.compat.v1.Session() as sess:

  a = tf.constant(5.0)
  b = tf.constant(6.0)
  c = tf.multiply(a, b)

  result = sess.run(c)
  print(result)