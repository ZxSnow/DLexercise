import tensorflow as tf


def tensorflow_demo():
    a = tf.constant(2)
    b = tf.constant(3)
    c = a + b
    print(c)
    # with tf.compat.v1.Session() as sess:
    #     a = tf.constant(2)
    #     b = tf.constant(3)
    #     c = a + b
    #     c_value = sess.run(c)
    #     print("再看看:", c_value)

    return None


if __name__ == '__main__':
    tensorflow_demo()
