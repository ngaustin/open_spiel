"""
Module for History-RNN inspired by the Sketch-RNN Architecture proposed by Ha et al. 2017
"""


import math
import tensorflow.compat.v1 as tf
from open_spiel.python import simple_nets


class MixtureDensityNetwork(tf.Module):

  def __init__(self,
               state_size,
               action_size,
               hidden_sizes,
               num_mixtures=3,
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
    self._state_size = state_size 
    self._num_mixtures = num_mixtures

    super(MixtureDensityNetwork, self).__init__(name=name)

    # Define the RNN Cell 
    with self.name_scope:
        # Output layer has size (2 * state_size) * num_mixtures + num_mixtures
        # Each mixture gets a mean and standard deviation vector of size state_size and we need probabilities of each mixture as well
        self._feed_forward = simple_nets.MLP(input_size=state_size + action_size, hidden_sizes=hidden_sizes, output_size=(2 * self._state_size) * self._num_mixtures + self._num_mixtures)

  @tf.Module.with_name_scope
  def __call__(self, x):
    """ This call will assume we only want the resulting hidden states determined by the VAE """
    output = self._feed_forward(x)

    # Split the output by mean, standard deviation, and mixture weights 
    means = tf.reshape(output[, :self._state_size * self._num_mixtures], [-1, self._num_mixtures, self._state_size])
    standard_deviations = tf.reshape(output[, self._state_size * self._num_mixtures: 2*self._state_size*self._num_mixtures], [-1, self._num_mixtures, self._state_size])
    mixture_weights = output[, 2*self._state_size*self._num_mixtures:]

    # Apply ELU to standard deviation. Apply softmax to mixture weights 
    standard_deviation = tf.nn.elu(standard_deviation) + 1 + 1e-15 
    mixture_weights = tf.nn.softmax(mixture_weights, axis=1)

    return means, standard_deviations, mixture_weights
