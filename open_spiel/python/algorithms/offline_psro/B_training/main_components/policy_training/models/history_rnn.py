"""
Module for History-RNN inspired by the Sketch-RNN Architecture proposed by Ha et al. 2017
"""


import math
import tensorflow.compat.v1 as tf
from open_spiel.python import simple_nets


class HistoryRNN(tf.Module):

  def __init__(self,
               input_size,
               latent_size,
               activate_final=False,
               name=None):
    """Create the HistoryRNN

    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      output_size: (int) number of outputs
      activate_final: (bool) should final layer should include a ReLU
      name: (string): the name to give to this network
    """

    super(HistoryRNN, self).__init__(name=name)

    # Define the RNN Cell 
    with self.name_scope:
        self._encoder = simple_nets.Linear(in_size=input_size, output_size=latent_size)
        self._decoder_cell = tf.nn.rnn_cell.LSTMCell(units=latent_size)

  def generate_history(self, s):
      h = self._encoder(s)

  @tf.Module.with_name_scope
  def __call__(self, x):
    """ This call will assume we only want the resulting hidden states determined by the VAE """
    mean, logvar = self.encode(x)
    z = self.reparametrize(mean, logvar)
    reconstructed = self.decode(z)
    return reconstructed, mean, logvar, z

