import tensorflow as tf

print('GPU',tf.test.is_gpu_available())
config = tf.ConfigProto()

config.gpu_options.allow_growth=True

sess = tf.Session(config=config)
a = tf.constant(2.)
b = tf.constant(4.)

print(a * b)