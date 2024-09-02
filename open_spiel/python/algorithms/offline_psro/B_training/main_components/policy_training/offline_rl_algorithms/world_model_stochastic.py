"""
This file is dedicated to training a feed-forward world model using a Mixture Density Network
"""

import collections
import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import math 
import time

from open_spiel.python import simple_nets
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.models.mdn import MixtureDensityNetwork

class WorldModel:
    def __init__(self, state_size, action_size, num_mixtures, hidden_sizes, model_args={}):
        self._state_size = state_size
        self._action_size = action_size
        self._num_mixtures = num_mixtures 
        self._hidden_sizes = hidden_sizes
        self._mixture_density_network = MixtureDensityNetwork(state_size, action_size, hidden_sizes, num_mixtures)

        self._session = model_args["session"]
        self._data = model_args["data"]
        self._batch_size = model_args["batch_size"]
        self._lr = model_args["learning_rate"]

        self._state_ph = tf.placeholder(shape=[None, self._state_size], dtype=tf.float32, name="state_ph")
        self._action_ph = tf.placeholder(shape=[None, 1], dtype=tf.int32, name="action_ph")
        self._next_state_ph = tf.placeholder(shape=[None, self._state_size], dtype=tf.float32, name="next_state_ph")
        
        delta_state = self._next_state_ph - self._state_ph
        network_input = tf.concat([self._state_ph, self._action_ph], axis=1)

        self._means, self._std_devs, self._mixture_probs = self._mixture_density_network(network_input)

        # self._means/self._std_devs shape: [N, num_mixtures, state_size]
        # self._mixture_probs shape: [N, num_mixtures]

        # Get the log probability of each point by summing across state_size the loglikelihood (independence assumption)
        log_prob_points = -.5 * tf.math.log(2 * math.pi) - tf.math.log(self._std_devs) - .5 * tf.math.square((delta_state - self._means) / self._std_devs)
        log_prob_points = tf.reduce_sum(log_prob_points, axis=2) # [N, num_mixtures]

        # Get the log probability of each mixture 
        log_mixture_probs = tf.math.log(self._mixture_probs + 1e-15)  # [N, num_mixtures]

        # Sum the two together and exponentiate
        sum_point_mixture_log_prob = tf.math.exp(log_prob_points + log_mixture_probs)  # [N, num_mixtures]

        # Sum across all mixtures and take negative log
        negative_log_likelihoods = -1 * tf.math.log(tf.reduce_sum(sum_point_mixture_log_prob, axis=1)) # [N]

        # Average is the loss
        self._loss = tf.reduce_mean(negative_log_likelihoods)

        # Optimizers
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
        self._train_step = self._optimizer.minimize(self._loss)

        self._initialize()


    def _initialize(self):
        # Initialize policy network, both value networks, all three target networks, and both optimizers
        init_model = tf.group(*[var.initializer for var in self._mixture_density_network.variables[:]])
        init_opt = tf.group(*[var.initializer for var in self._optimizer.variables()])

        self._session.run(tf.group(*[init_model, init_opt]))
        return         

    def get_next_state(self, state):
    
        self._means, self._std_devs, self._mixture_probs = self._session.run(
            [self._means, self._std_devs, self._mixture_probs], 
            feed_dict={
                self._state_ph: states,
                self._action_ph: actions,
                self._gumbel_noise_ph:gumbel1,
                self._state_mean_ph:[self._state_mean],
                self._state_std_ph:[self._state_std]
            }
        )

        # Sample from mixtures 
        mixture_choice = np.random.choice(self._num_mixtures, p=self._mixture_probs)
        mean = self._means[: ,mixture_choice, :]
        std_dev = self._std_devs[:, mixture_choice, :]

        # Sample from marginals 
        dist = tf.distributions.Normal(loc=mean, scale=std_dev)
        sample = dist.sample([1]) # [1 x self._state_size]

        # Add to original state
        next_state = state + sample

        # Return 
        return next_state 
    
    def train(self, num_gradient_steps):
        self._state_ph = tf.placeholder(shape=[None, self._state_size], dtype=tf.float32, name="state_ph")
        self._action_ph = tf.placeholder(shape=[None, 1], dtype=tf.int32, name="action_ph")
        self._next_state_ph

        data_size = len(self._data)
        for _ in range(num_gradient_steps):
            indices = np.random.choice(data_size, self._batch_size)

            states = []
            actions = []
            next_states = []

            for i in indices:
                states.append(self._data[i].global_state)
                actions.append(self._data[i].actions)
                next_states.append(self._data[i].next_global_state)
            
            loss, _ = self._session.run(
                [self._loss, self._train_step], 
                feed_dict={
                    self._state_ph: states,
                    self._action_ph: actions,
                    self._next_state_ph: next_states,
                }
            )
        return 


