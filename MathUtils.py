import tensorflow as tf
import numpy as np
# Initialize and define metasurfaces: need split_complex and complex_initializer_random
def split_complex(c):
    return tf.math.real(c), tf.math.imag(c)
# element-wise multiply complex tensors `m1` and `m2`
def complex_mul(m1_real, m1_imag, m2_real, m2_imag):
    return m1_real * m2_real - m1_imag * m2_imag, m1_real * m2_imag + m1_imag * m2_real
def complex_initializer_random(shape):
    """
    Generates a complex valued tensor with values that are distributed with uniform density in the unit circle of the
    complex plane
    """
    return np.sqrt(np.random.rand(*shape)) * np.exp(1j * np.random.rand(*shape) * np.pi * 2)