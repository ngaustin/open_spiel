"""
Module for Variational Auto-Encoder (VAE)
"""


import math
import tensorflow.compat.v1 as tf
from open_spiel.python import simple_nets


class SequentialVAE(tf.Module):

  def __init__(self,
               input_size,
               latent_size,
               activate_final=False,
               name=None):
    """Create the VAE

    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      output_size: (int) number of outputs
      activate_final: (bool) should final layer should include a ReLU
      name: (string): the name to give to this network
    """

    super(SequentialVAE, self).__init__(name=name)

    # Define the RNN Cell 
    with self.name_scope:
        self._encoder_cell = tf.nn.rnn_cell.LSTMCell(units=input_size)

        self._mean_variance_layer = simple_nets.Linear(in_size=input_size, out_size=latent_size * 2, activate_relu=False)

        self._decoder_cell = tf.nn.rnn_cell.LSTMCell(units=latent_size)

  def encode(self, x):
    val, state = tf.nn.dynamic_run(self._encoder_cell, x, dtype=tf.float32)
    val = self._mean_variance_layer(val)
    mean, logvar = tf.split(val, num_or_size_splits=2, axis=)
    return mean, logvar

  def reparametrize(self, mean, logvar):
    epsilon = tf.random.normal(shape=mean.shape)
    return epsilon * tf.exp(logvar * .5) + mean

  def decode(self, z):
    return self._decoder_cell(z)

  @tf.Module.with_name_scope
  def __call__(self, x):
    """ This call will assume we only want the resulting hidden states determined by the VAE """
    mean, logvar = self.encode(x)
    z = self.reparametrize(mean, logvar)
    reconstructed = self.decode(z)
    return reconstructed, mean, logvar, z

