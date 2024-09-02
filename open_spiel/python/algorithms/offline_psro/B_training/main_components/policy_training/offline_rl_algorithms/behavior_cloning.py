"""
This file is dedicated to a behavior cloning implemenetation. It will be used to estimate the behavior policy that generates a dataset.
"""

import collections
import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import math 
import time

from open_spiel.python import simple_nets


class JointBehaviorCloning:
    def __init__(self, state_size, action_size, session, hidden_sizes, batch_size, learning_rate, num_actions):
        self._state_size = state_size
        self._action_size = action_size 
        self._session = session
        self._hidden_sizes = hidden_sizes
        self._batch_size = batch_size
        self._lr = learning_rate
        self._num_actions = num_actions
        
        self._network = simple_nets.MLP(input_size=self._state_size, hidden_sizes=self._hidden_sizes, output_size=self._num_actions)

        self._state_ph = tf.placeholder(shape=[None, self._state_size], dtype=tf.float32, name="state_ph")
        self._action_ph = tf.placeholder(shape=[None], dtype=tf.int32, name="action_ph")

        self._log_probs = self._network(self._state_ph)
        loss_class = tf.losses.softmax_cross_entropy

        # Convert the actions to one hot vectors
        self.one_hot_vectors = tf.one_hot(self._action_ph, depth=self._num_actions)

        # Plug into cross entropy class
        self._loss = tf.reduce_mean(loss_class(self.one_hot_vectors, self._log_probs))# weights=tf.math.exp(self._return_ph)))

        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
        self._learn_step = self._optimizer.minimize(self._loss)

        self.initialize()

    def initialize(self):
        variable_list = []
        init_opt = tf.group(*[var.initializer for var in self._optimizer.variables()])
        variable_list.append(init_opt)
        variable_list.append(tf.group(*[var.initializer for var in self._network.variables[:]]))

        self._session.run(tf.group(*variable_list))

    def probability(self, state, action):

        log_probs = self._session.run(
            [self._log_probs],
            feed_dict={
                self._state_ph: [state] 
            }
        )[0]
        probs = (np.exp(log_probs) / np.sum(np.exp(log_probs)))[0]
        return probs[action]

    def train(self, data, steps):
        datapoints = []
        for trajectory in data:
            datapoints.extend(trajectory)
        num_datapoints = len(datapoints)

        losses = []
        for _ in range(steps):
            indices = np.random.choice(num_datapoints, self._batch_size)

            states = []
            actions = []
            for i in indices:
                states.append(datapoints[i].global_state)
                action = datapoints[i].actions[datapoints[i].relevant_players[0]]
                # Assume only one player is acting at any given time
                actions.append(action)

            loss, _, log_probs = self._session.run(
                [self._loss, self._learn_step, self._log_probs],
                feed_dict={
                    self._state_ph: states,
                    self._action_ph: actions,
                })
            losses.append(loss)
        return losses
