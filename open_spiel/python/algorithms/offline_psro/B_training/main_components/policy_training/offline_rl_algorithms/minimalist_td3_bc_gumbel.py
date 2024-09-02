"""
Main wrapper responsible for using Behavior Proximal Policy Optimization to iteratively train a best response 
using a fixed dataset. Actor-critic framework so that value function remains accurate. Furthermore, BPPO is an
offline algorithm that works well for discrete action spaces and is simple to implement.
"""

import collections
import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import time

from open_spiel.python import simple_nets
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.policy_wrapper import PolicyWrapper

class TD3_BC_Gumbel(PolicyWrapper):
    
    def __init__(self, num_actions, state_size, policy_args={}):
        super().__init__(self, num_actions, state_size) 

        self._data = policy_args["data"]
        self._session = policy_args["session"]
        self._policy_network_lr = policy_args["policy_network_lr"]
        self._value_network_lr = policy_args["value_network_lr"]
        self._policy_network_shape = policy_args["policy_network_shape"]
        self._value_network_shape = policy_args["value_network_shape"]
 
        self._discount = policy_args["discount"]
        self._batch_size = policy_args["batch_size"]
        self._temp = policy_args["temp"]
        self._alpha = policy_args["alpha"]
        self._tau = policy_args["tau"]
        self._policy_update_frequency = policy_args["policy_update_frequency"]

        self._state_mean = np.mean([t.info_state for t in self._data], axis=0)
        self._state_std = np.std([t.info_state for t in self._data], axis=0)
        
        # Placeholders 
        self._state_ph = tf.placeholder(shape=[None, self._state_size], dtype=tf.float32, name="state_ph")
        self._action_ph = tf.placeholder(shape=[None, 1], dtype=tf.int32, name="action_ph")
        self._reward_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="reward_ph")
        self._done_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="done_ph")
        self._next_state_ph = tf.placeholder(shape=[None, self._state_size], dtype=tf.float32, name="next_state_ph")
        self._next_legal_actions_mask_ph = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32, name="next_legal_actions_mask_ph")
        self._gumbel_noise_ph = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32, name="gumbel_noise_ph")
        self._gumbel_noise_2_ph = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32, name="gumbel_noise_2_ph")
        self._state_mean_ph = tf.placeholder(shape=[None, self._state_size], dtype=tf.float32, name="state_mean_ph")
        
        ######### Minimalist TD3 + BC with Gumbel-Softmax Trick ##########
        # NOTE: For TD3, the input to the Q-network is state AND action, outputting one Q-value
        self._policy_network = simple_nets.MLP(input_size=self._state_size, hidden_sizes=self._policy_network_shape, output_size=self._num_actions)
        self._q1 = simple_nets.MLP(input_size=self._state_size + self._num_actions, hidden_sizes=self._value_network_shape, output_size=1)
        self._q2 = simple_nets.MLP(input_size=self._state_size + self._num_actions, hidden_sizes=self._value_network_shape, output_size=1)

        self._policy_target = simple_nets.MLP(input_size=self._state_size, hidden_sizes=self._policy_network_shape, output_size=self._num_actions)
        self._q1_target = simple_nets.MLP(input_size=self._state_size + self._num_actions, hidden_sizes=self._value_network_shape, output_size=1)
        self._q2_target = simple_nets.MLP(input_size=self._state_size + self._num_actions, hidden_sizes=self._value_network_shape, output_size=1)

        ########## Copy ############
        self._copy_q1 = self._create_network_copy_op(self._q1, self._q1_target)
        self._copy_q2 = self._create_network_copy_op(self._q2, self._q2_target)
        self._copy_policy = self._create_network_copy_op(self._policy_network, self._policy_target)

        ######### Normalize Everything ##########
        state_input = (self._state_ph - self._state_mean_ph) / (self._state_std + 1e-8) 
        next_state_input = (self._next_state_ph - self._state_mean_ph) / (self._state_std + 1e-8)

        ######## Value Update ############
        target_actions = self._policy_target(next_state_input) # [N x self._num_actions] of logits

        # NOTE: We apply a LARGE negative value to logits corresponding to invalid actions
        target_actions = target_actions + (1 - self._next_legal_actions_mask_ph) * (-1e9)
        # self._target_actions_before_gumbel = target_actions
        # self._target_actions_exp = tf.math.exp((self._gumbel_noise_ph + target_actions) / self._temp)


        # NOTE: We are applying the gumbel technique to INPUT actions to the network as well! (Similar to gaussian noise in original TD3)
        self._target_actions = self._apply_gumbel(target_actions, self._gumbel_noise_ph, self._temp)

        q1_target = self._q1_target(tf.concat([next_state_input, self._target_actions], 1))
        q2_target = self._q2_target(tf.concat([next_state_input, self._target_actions], 1))
        min_q_target = tf.math.minimum(q1_target, q2_target)

        q_value_target = tf.stop_gradient(self._reward_ph + self._discount * (1 - self._done_ph) * min_q_target)

        # TODO: Convert the self._action_ph into things that look similar to gumbel noised inputs
        action_input = tf.one_hot(tf.squeeze(self._action_ph), depth=self._num_actions)
        action_input_gumbel = self._apply_gumbel(action_input, self._gumbel_noise_2_ph, self._temp)

        self._q_loss = tf.reduce_mean(tf.square(self._q1(tf.concat([state_input, action_input_gumbel], 1)) - q_value_target))
        self._q_loss += tf.reduce_mean(tf.square(self._q2(tf.concat([state_input, action_input_gumbel], 1)) - q_value_target))

        self._q_optimizer = tf.train.AdamOptimizer(learning_rate=self._value_network_lr)
        self._q_train_step = self._q_optimizer.minimize(self._q_loss)

        ######## Policy Update ###########
        self._actions = self._policy_network(state_input)
        actions_gumbel = self._apply_gumbel(self._actions, self._gumbel_noise_ph, self._temp)

        q_values = self._q1(tf.concat([state_input, actions_gumbel], 1))

        # NOTE: Action_input is a bunch of one-hot vectors
        behavior_cloning_loss = tf.square(actions_gumbel - action_input)
        lam = self._alpha / (tf.reduce_mean(tf.math.abs(q_values)))
        self._policy_loss = -1 * tf.reduce_mean(lam * q_values - behavior_cloning_loss)

        self._policy_optimizer = tf.train.AdamOptimizer(learning_rate=self._value_network_lr)
        self._policy_step = self._policy_optimizer.minimize(self._policy_loss, var_list=self._policy_network.variables[:])


        ######### Target Updates #########
        self._update_q1 = self._create_network_update_op(self._q1, self._q1_target)
        self._update_q2 = self._create_network_update_op(self._q2, self._q2_target)
        self._update_policy = self._create_network_update_op(self._policy_network, self._policy_target)

        # Initialize
        self._initialize()

        # Copy the target networks over
        self._session.run(self._copy_q1)
        self._session.run(self._copy_q2)
        self._session.run(self._copy_policy)

    def _initialize(self):
        # Initialize policy network, both value networks, all three target networks, and both optimizers
        init_policy = tf.group(*[var.initializer for var in self._policy_network.variables[:]])
        init_q1 = tf.group(*[var.initializer for var in self._q1.variables[:]])
        init_q2 = tf.group(*[var.initializer for var in self._q2.variables[:]])

        init_policy_target = tf.group(*[var.initializer for var in self._policy_target.variables[:]])
        init_q1_target = tf.group(*[var.initializer for var in self._q1_target.variables[:]])
        init_q2_target = tf.group(*[var.initializer for var in self._q2_target.variables[:]])

        init_opt_policy = tf.group(*[var.initializer for var in self._policy_optimizer.variables()])
        init_opt_q = tf.group(*[var.initializer for var in self._q_optimizer.variables()])

        self._session.run(tf.group(*[init_policy, init_q1, init_q2,
                                    init_policy_target, init_q1_target, init_q2_target,
                                    init_opt_policy, init_opt_q]))
        
        return         

    def get_policy_output_variable_name(self):
        name = self._actions.name 
        if ":" in name:
            name = name[:name.index(":")]
        return name

    def _apply_gumbel(self, logits, gumbel_samples, temp):
        # Assuming that all of these are [N x self._num_actions]
        exponents = tf.math.exp((gumbel_samples + logits) / temp)

        return exponents / (tf.expand_dims(tf.reduce_sum(exponents, axis=1), axis=1) + 1e-9)

    def _create_network_copy_op(self, source_network, target_network):

        return tf.group([
            tf.assign(target_v, v)
            for (target_v, v) in zip(target_network.variables[:], source_network.variables[:])
        ])

    def _create_network_update_op(self, source_network, target_network):

        return tf.group([
            tf.assign(target_v, (1 - self._tau) * v + self._tau * target_v)
            for (target_v, v) in zip(target_network.variables[:], source_network.variables[:])
        ])

    def step(self, state, legal_actions):
        logits = self._session.run(
            [self._actions],
            feed_dict={
                self._state_ph: [state],
                self._state_mean_ph: [self._state_mean]
            })[0][0]

        logits_legal = [logits[i] if i in legal_actions else -1e5 for i in range(self._num_actions)]
        # print("legal", logits, logits_legal, legal_actions)
        return logits_legal.index(max(logits_legal))

    def train(self, num_gradient_steps):
        # We assume the data is structured as a list of Transition objects, not as a list of list of Transition objects (not trajectories)
        data_size = len(self._data)

        

        for i in range(num_gradient_steps):
        # For however many updates 
            # Get a batch of samples from data 
            indices = np.random.choice(data_size, self._batch_size)
            
            # Get two samples from gumbel distribution 
            gumbel1 = np.random.gumbel(size=[self._batch_size, self._num_actions])
            gumbel2 = np.random.gumbel(size=[self._batch_size, self._num_actions])

            states = []
            actions = []
            rewards = []
            dones = []
            next_states = []
            next_legal_action_masks = []

            for i in indices:
                states.append(self._data[i].info_state)
                actions.append([self._data[i].action])
                rewards.append([self._data[i].reward])
                dones.append([self._data[i].done])
                next_states.append(self._data[i].next_info_state)
                next_legal_action_masks.append(self._data[i].next_legal_actions_mask)
            
            # Do a value step 
            loss, _ = self._session.run(
                [self._q_loss, self._q_train_step], 
                feed_dict={
                    self._state_ph: states,
                    self._action_ph: actions,
                    self._reward_ph: rewards, 
                    self._done_ph: dones, 
                    self._next_state_ph: next_states,
                    self._next_legal_actions_mask_ph: next_legal_action_masks,
                    self._gumbel_noise_ph:gumbel1,
                    self._gumbel_noise_2_ph:gumbel2,
                    self._state_mean_ph:[self._state_mean]
                }
            )
            # print("Q_Loss: ", loss)
            """
            print("Gumbel: ", gumbel1)
            print('')
            print("Dones:", dones)
            print('')
            print("Before: ", before, exp)
            print('')"""

            # If policy step is in the works 
            if i % self._policy_update_frequency:
                loss, _ = self._session.run(
                    [self._policy_loss, self._policy_step],
                    feed_dict={
                        self._state_ph:states,
                        self._action_ph:actions,
                        self._state_mean_ph:[self._state_mean],
                        self._gumbel_noise_ph:gumbel1
                    }
                )

                # print("policy loss: ", loss)

                # Update the target networks if updated policies
                self._session.run(self._update_q1)
                self._session.run(self._update_q2)
                self._session.run(self._update_policy)
                
        return

    def probabilities(self, state, legal_actions_mask, numpy=False):
        return 
 
    def probabilities_with_actions(self, state, action, legal_actions_mask, numpy=False):
        return 

    @property
    def num_actions(self):
        return self._num_actions
    
    @property 
    def state_size(self):
        return self._state_size

    @property
    def id(self):
        return self._id
        
