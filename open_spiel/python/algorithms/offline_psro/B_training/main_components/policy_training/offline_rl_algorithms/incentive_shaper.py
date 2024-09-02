"""
This is just a simple state-action to reward model
"""

import collections
import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import math 
import time

from open_spiel.python import simple_nets


class IncentiveShaper:
    def __init__(self, state_size, num_actions, num_players, session, hidden_sizes=[250, 250], learning_rate=3e-4):
        self._state_size = state_size 
        self._num_actions = num_actions 
        self._hidden_sizes = hidden_sizes 
        self._lr = learning_rate
        self._num_players = num_players 
        self._session = session

        self._is_frozen = False 
        self._frozen_graph = None

        self._model = simple_nets.MLP(input_size=self._state_size + self._num_actions + self._num_players, hidden_sizes=self._hidden_sizes, output_size=1)

        self._state_ph = tf.placeholder(shape=[None, self._state_size], dtype=tf.float32, name="state_ph")
        self._action_ph = tf.placeholder(shape=[None, 1], dtype=tf.int32, name="action_ph")
        self._player_ph = tf.placeholder(shape=[None, 1], dtype=tf.int32, name="player_ph")
        self._reward_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="reward_ph")

        action_one_hot_vectors = tf.squeeze(tf.one_hot(self._action_ph, depth=self._num_actions, axis=-1), axis=1)
        player_one_hot_vectors = tf.squeeze(tf.one_hot(self._player_ph, depth=self._num_players, axis=-1), axis=1)
        network_input = tf.concat([self._state_ph, action_one_hot_vectors, player_one_hot_vectors], axis=1)  # [N, self._state_size + self._num_actions + self._num_players]

        self._model_reward = self._model(network_input)  # [N , 1]
        self._loss = tf.reduce_mean(tf.square(self._model_reward - self._reward_ph))

        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
        self._train_step = self._optimizer.minimize(self._loss)

        self.initialize()
        return 

    def initialize(self):
        variable_list = []

        init_model = tf.group(*[var.initializer for var in self._model.variables[:]])
        variable_list.append(init_model)

        init_opt = tf.group(*[var.initializer for var in self._optimizer.variables()])
        variable_list.append(init_opt)

        self._session.run(tf.group(*variable_list))      
    
    def one_step_learn(self, states, action, reward, player):
        loss, _ = self._session.run(
                [self._loss, self._train_step], 
                feed_dict={
                    self._state_ph: states,
                    self._action_ph: action,
                    self._reward_ph: reward,
                    self._player_ph: player
                }
            )
        return loss 

    def predict(self, state, action, player, frozen_session=None):
        state = np.reshape(state, [1, -1])
        action = np.reshape(action, [1, -1])
        player = np.reshape(player, [1, -1])

        if self._is_frozen:
            state_input = self._frozen_graph.get_tensor_by_name(self._frozen_input_variables["state_ph"])
            action_input = self._frozen_graph.get_tensor_by_name(self._frozen_input_variables["action_ph"])
            player_input = self._frozen_graph.get_tensor_by_name(self._frozen_input_variables["player_ph"])
            reward_output = self._frozen_graph.get_tensor_by_name(self._frozen_output_variables['model_reward'])
            reward = frozen_session.run([reward_output], feed_dict={state_input: state, action_input: action, player_input: player})[0]
        else:
            reward = self._session.run(
                    [self._model_reward], 
                    feed_dict={
                        self._state_ph: state,
                        self._action_ph: action,
                        self._player_ph: player
                    }
                )[0]
        return reward[0]

    def get_output_variable_names(self):
        output_variables = [self._model_reward]
        names = [var.name for var in output_variables]

        # self._frozen_output_variables should map to the tensors themselves, so we want the ':0'
        self._frozen_output_variables = {"model_reward": names[0]}
        self._frozen_input_variables = {"state_ph": self._state_ph.name, "action_ph": self._action_ph.name, "player_ph": self._player_ph.name}

        # For the names, we want only the operation. So, we get rid of the ':0' at the end
        names = [name[:name.index(":")] if ":" in name else name for i, name in enumerate(names)]
        return names

    def get_frozen_graph(self):
        return self._frozen_graph

    def freeze(self, model_manager, save_path):
        model_manager.freeze_graph(save_path, "incentive_model_frozen.pb", self.get_output_variable_names())
        frozen_graph = model_manager.load_frozen_graph(save_path, "incentive_model_frozen.pb")
        self._is_frozen = True 
        self._frozen_graph = frozen_graph
        print("Incentive model is frozen.")
        print("Incentive model info:     Input variables: {}      Output variables: {} \n".format(self._frozen_input_variables, self._frozen_output_variables))
        # print("Operations: ", [op.name for op in self._frozen_graph.get_operations()])
        