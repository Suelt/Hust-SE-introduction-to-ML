import tensorflow as tf
tf.compat.v1.disable_eager_execution()　＃保证sess.run()能够正常运行
hello = tf.constant('hello,tensorflow')
sess= tf.compat.v1.Session()＃版本2.0的函数
print(sess.run(hello))
