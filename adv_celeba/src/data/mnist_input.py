import tensorflow as tf
import numpy as np

def distorted_inputs(data_dir, batch_size):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    images, label_batch =  tf.train.shuffle_batch( \
            [train_data, train_labels],
            batch_size=batch_size,
            num_threads=16,
            capacity=4000+3*batch_size,
            min_after_dequeue=4000,
            enqueue_many=True)
    return tf.reshape(images, [batch_size, 28, 28, 1]), \
            tf.reshape(label_batch, [batch_size])

def inputs(eval_data, data_dir, batch_size):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    src = mnist.train
    if eval_data:
        src = mnist.test
    train_data = src.images
    train_labels = np.asarray(src.labels, dtype=np.int32)
    images, label_batch =  tf.train.shuffle_batch( \
            [train_data, train_labels],
            batch_size=batch_size,
            num_threads=16,
            capacity=4000+3*batch_size,
            min_after_dequeue=4000,
            enqueue_many=True)
    return tf.reshape(images, [batch_size, 28, 28, 1]), \
            tf.reshape(label_batch, [batch_size])

