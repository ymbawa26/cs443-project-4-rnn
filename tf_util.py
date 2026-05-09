'''tf_util.py
A few helper functions provided to you for various projects in CS 443.
Oliver W. Layton
CS 443: Bio-Inspired Machine Learning
'''
import tensorflow as tf


def arange_index(x, y):
    '''Reproduces arange indexing from NumPy in TensorFlow.'''
    rows = tf.range(len(x))
    rc_tuples = tf.stack([rows, y], axis=1)
    return tf.gather_nd(x, rc_tuples)
