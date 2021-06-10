import tensorflow as tf


# 使用TensorFlow输出Hello

# 创建一个常量操作( Constant op )
# 这个 op 会被作为一个节点( node )添加到默认计算图上.
#
# 该构造函数返回的值就是常量节点(Constant op)的输出.
hello = tf.constant('Hello, world!')

# 启动TensorFlow会话
sess = tf.compat.v1.Session()

# 运行 hello 节点
print(sess.run(hello))
sess.close()