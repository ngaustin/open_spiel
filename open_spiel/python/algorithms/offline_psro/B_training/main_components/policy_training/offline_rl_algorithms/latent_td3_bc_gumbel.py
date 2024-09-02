"""
Main wrapper for VAE+TD3+BC+Gumbel+COP-TD Target Resampling
"""

import collections
import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import time

from open_spiel.python import simple_nets
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.policy_wrapper import PolicyWrapper
from open_spiel.python.algorithms.offline_psro.utils.utils import compute_hash_string

class Latent_TD3_BC_Gumbel(PolicyWrapper):
    
    def __init__(self, num_actions, state_size, policy_args={}):
        super().__init__(self, num_actions, state_size) 

        self._data = policy_args["data"]
        self._session = policy_args["session"]
        self._policy_network_lr = policy_args["policy_network_lr"]
        self._value_network_lr = policy_args["value_network_lr"]
        self._policy_network_shape = policy_args["policy_network_shape"]
        self._value_network_shape = policy_args["value_network_shape"]
        self._encoder_network_shape = policy_args["encoder_network_shape"]
        self._latent_space_size = policy_args["latent_space_size"]
        self._triplet_loss_epsilon = policy_args["triplet_loss_epsilon"]
        self._contrastive_loss_weight = policy_args["contrastive_loss_weight"]
 
        self._discount = policy_args["discount"]
        self._batch_size = policy_args["batch_size"]
        self._temp = policy_args["temp"]
        self._alpha = policy_args["alpha"]
        self._tau = policy_args["tau"]
        self._policy_update_frequency = policy_args["policy_update_frequency"]

        self._state_mean = np.mean([t.info_state for t in self._data], axis=0)
        self._state_std = np.std([t.info_state for t in self._data], axis=0)

        # Analyze the true states for overlap
        state_hash_to_index = {}
        self._data_to_state_index = {}
        for i, d in enumerate(self._data):
            curr_hash = compute_hash_string(d.global_state)
            if curr_hash not in state_hash_to_index:
                state_hash_to_index[curr_hash] = len(state_hash_to_index)
            self._data_to_state_index[i] = state_hash_to_index[curr_hash]
            
        # Keep a dictionary of true_state to index. When adding, assign the index to the current length of the dictionary.
        # Create a dictionary mapping data index to true_state index
        
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
        self._state_std_ph = tf.placeholder(shape=[None, self._state_size], dtype=tf.float32, name="state_std_ph")

        self._anchor_ph = tf.placeholder(shape=[None], dtype=tf.int32, name="anchor")
        self._positive_ph = tf.placeholder(shape=[None], dtype=tf.int32, name="positive")
        self._negative_ph = tf.placeholder(shape=[None], dtype=tf.int32, name="negative")
        self._has_triplet = tf.placeholder(shape=(), dtype=tf.bool, name="has_triplet")

        ######### MLP Encoder Section for Hidden State POMDP Inference ###############

        self._encoder = simple_nets.MLP(input_size=self._state_size, hidden_sizes=self._encoder_network_shape, output_size=self._latent_space_size, activate_final=True)
        self._encoder_target = simple_nets.MLP(input_size=self._state_size, hidden_sizes=self._encoder_network_shape, output_size=self._latent_space_size, activate_final=True)


        ######### Minimalist TD3 + BC with Gumbel-Softmax Trick ##########
        # NOTE: For TD3, the input to the Q-network is state AND action, outputting one Q-value
        self._policy_network = simple_nets.MLP(input_size=self._state_size, hidden_sizes=self._policy_network_shape, output_size=self._num_actions)
        self._q1 = simple_nets.MLP(input_size=self._latent_space_size + self._num_actions, hidden_sizes=self._value_network_shape, output_size=1)
        self._q2 = simple_nets.MLP(input_size=self._latent_space_size + self._num_actions, hidden_sizes=self._value_network_shape, output_size=1)

        self._policy_target = simple_nets.MLP(input_size=self._state_size, hidden_sizes=self._policy_network_shape, output_size=self._num_actions)
        self._q1_target = simple_nets.MLP(input_size=self._latent_space_size + self._num_actions, hidden_sizes=self._value_network_shape, output_size=1)
        self._q2_target = simple_nets.MLP(input_size=self._latent_space_size + self._num_actions, hidden_sizes=self._value_network_shape, output_size=1)

        ########## Copy ############
        self._copy_q1 = self._create_network_copy_op(self._q1, self._q1_target)
        self._copy_q2 = self._create_network_copy_op(self._q2, self._q2_target)
        self._copy_policy = self._create_network_copy_op(self._policy_network, self._policy_target)
        self._copy_encoder = self._create_network_copy_op(self._encoder, self._encoder_target)

        ######### Normalize Everything ##########
        state_input = (self._state_ph - self._state_mean_ph) / (self._state_std_ph + 1e-8)
        next_state_input = (self._next_state_ph - self._state_mean_ph) / (self._state_std_ph + 1e-8)

        ######## Encoder Update ########### 
        # NOTE: Keep in mind whether we use the target encoder or current encoder for z depends whether we are querying the target or current Q/policy networks! 
        z_state = self._encoder(state_input)
        z_next_state_target = self._encoder_target(next_state_input)

        # Contrastive loss. Triplet for simplicity!

        anchor_states = tf.gather(z_state, self._anchor_ph)
        positive_states = tf.gather(z_state, self._positive_ph)
        negative_states = tf.gather(z_state, self._negative_ph)

        positive_distance = tf.reduce_sum(tf.square(anchor_states - positive_states), axis=1)  # Num_pairs x 1
        negative_distance = tf.reduce_sum(tf.square(anchor_states - negative_states), axis=1)  

        self._contrastive_loss = tf.reduce_mean(tf.clip_by_value(positive_distance - negative_distance + self._triplet_loss_epsilon, 0, np.inf))
        

        ######## Value Update ############
        target_actions = self._policy_target(next_state_input) # [N x self._num_actions] of logits

        # NOTE: We apply a LARGE negative value to logits corresponding to invalid actions
        target_actions = target_actions + (1 - self._next_legal_actions_mask_ph) * (-1e9)

        # NOTE: We are applying the gumbel technique to INPUT actions to the network as well! (Similar to gaussian noise in original TD3)
        self._target_actions = self._apply_gumbel(target_actions, self._gumbel_noise_ph, self._temp)

        q1_target = self._q1_target(tf.concat([z_next_state_target, self._target_actions], 1))
        q2_target = self._q2_target(tf.concat([z_next_state_target, self._target_actions], 1))
        min_q_target = tf.math.minimum(q1_target, q2_target)

        q_value_target = tf.stop_gradient(self._reward_ph + self._discount * (1 - self._done_ph) * min_q_target)

        # TODO: Convert the self._action_ph into things that look similar to gumbel noised inputs
        action_input = tf.one_hot(tf.squeeze(self._action_ph), depth=self._num_actions)
        action_input_gumbel = self._apply_gumbel(action_input, self._gumbel_noise_2_ph, self._temp)

        self._q_loss = tf.reduce_mean(tf.square(self._q1(tf.concat([z_state, action_input_gumbel], 1)) - q_value_target))
        self._q_loss += tf.reduce_mean(tf.square(self._q2(tf.concat([z_state, action_input_gumbel], 1)) - q_value_target))

        self._q_optimizer = tf.train.AdamOptimizer(learning_rate=self._value_network_lr)
        self._q_train_step = self._q_optimizer.minimize(self._q_loss + self._contrastive_loss * self._contrastive_loss_weight)

        ######## Policy Update ###########
        self._actions = self._policy_network(state_input)  # TODO: Remove this. Was z_state before!  
        actions_gumbel = self._apply_gumbel(self._actions, self._gumbel_noise_ph, self._temp)

        q_values = self._q1(tf.concat([tf.stop_gradient(z_state), actions_gumbel], 1))

        # NOTE: Action_input is a bunch of one-hot vectors
        self._behavior_cloning_loss = tf.square(actions_gumbel - action_input)
        lam = self._alpha / (tf.reduce_mean(tf.math.abs(q_values)))
        self._policy_loss = -1 * tf.reduce_mean(lam * q_values - self._behavior_cloning_loss)

        self._behavior_cloning_loss = tf.reduce_mean(self._behavior_cloning_loss)

        self._policy_optimizer = tf.train.AdamOptimizer(learning_rate=self._value_network_lr)
        self._policy_step = self._policy_optimizer.minimize(self._policy_loss, var_list=self._policy_network.variables[:]+self._encoder.variables[:])
        self._bc_step = self._policy_optimizer.minimize(self._behavior_cloning_loss, var_list=self._policy_network.variables[:])

        ######### Target Updates #########
        self._update_q1 = self._create_network_update_op(self._q1, self._q1_target)
        self._update_q2 = self._create_network_update_op(self._q2, self._q2_target)
        self._update_policy = self._create_network_update_op(self._policy_network, self._policy_target)
        self._update_encoder = self._create_network_update_op(self._encoder, self._encoder_target)
        
        ######### Initialize ##############
        self._initialize()

        ######## Copy Target Nets #########
        self._session.run(self._copy_q1)
        self._session.run(self._copy_q2)
        self._session.run(self._copy_policy)
        self._session.run(self._copy_encoder)

    def _initialize(self):
        # Initialize policy network, both value networks, all three target networks, and both optimizers
        init_policy = tf.group(*[var.initializer for var in self._policy_network.variables[:]])
        init_q1 = tf.group(*[var.initializer for var in self._q1.variables[:]])
        init_q2 = tf.group(*[var.initializer for var in self._q2.variables[:]])
        init_encoder = tf.group(*[var.initializer for var in self._encoder.variables[:]])

        init_policy_target = tf.group(*[var.initializer for var in self._policy_target.variables[:]])
        init_q1_target = tf.group(*[var.initializer for var in self._q1_target.variables[:]])
        init_q2_target = tf.group(*[var.initializer for var in self._q2_target.variables[:]])
        init_encoder_target = tf.group(*[var.initializer for var in self._encoder_target.variables[:]])

        init_opt_policy = tf.group(*[var.initializer for var in self._policy_optimizer.variables()])
        init_opt_q = tf.group(*[var.initializer for var in self._q_optimizer.variables()])

        self._session.run(tf.group(*[init_policy, init_q1, init_q2, init_encoder,
                                    init_policy_target, init_q1_target, init_q2_target, init_encoder_target,
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
                self._state_mean_ph: [self._state_mean],
                self._state_std_ph: [self._state_std]
            })[0][0]

        logits_legal = [logits[i] if i in legal_actions else -1e9 for i in range(self._num_actions)]
        # print("legal", logits, logits_legal, legal_actions)
        return logits_legal.index(max(logits_legal))
    
    def pretrain_bc(self, num_gradient_steps):
        data_size = len(self._data)

        for i in range(num_gradient_steps):
            indices = np.random.choice(data_size, self._batch_size)

            gumbel1 = np.random.gumbel(size=[self._batch_size, self._num_actions])

            states = []
            actions = []
            
            for i in indices:
                states.append(self._data[i].info_state)
                actions.append([self._data[i].action])
            
            
            # Do a bc step 
            loss, _ = self._session.run(
                [self._behavior_cloning_loss, self._bc_step], 
                feed_dict={
                    self._state_ph: states,
                    self._action_ph: actions,
                    self._gumbel_noise_ph:gumbel1,
                    self._state_mean_ph:[self._state_mean],
                    self._state_std_ph:[self._state_std]
                }
            )

            print("BC Step Loss: ", loss)


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
            true_state_indices = []

            for i in indices:
                states.append(self._data[i].info_state)
                actions.append([self._data[i].action])
                rewards.append([self._data[i].reward])
                dones.append([self._data[i].done])
                next_states.append(self._data[i].next_info_state)
                next_legal_action_masks.append(self._data[i].next_legal_actions_mask)
                true_state_indices.append(self._data_to_state_index[i])
            
            anchors = []
            positives = []
            negatives = []

            for i in range(self._batch_size - 1):
                # Look for any index after i that is equal in true_state_indices[indices[i/j]]
                possible_negatives = [k for k, true_state_index in enumerate(true_state_indices) if true_state_index != true_state_indices[i]]
                for j in range(i+1, self._batch_size - 1):
                    if true_state_indices[i] == true_state_indices[j] and indices[i] != indices[j]:
                        anchors.append(i)
                        positives.append(j)
                        negatives.append(np.random.choice(possible_negatives))
            
            # Do a value step 
            loss, contrastive_loss, _ = self._session.run(
                [self._q_loss, self._contrastive_loss, self._q_train_step], 
                feed_dict={
                    self._state_ph: states,
                    self._action_ph: actions,
                    self._reward_ph: rewards, 
                    self._done_ph: dones, 
                    self._next_state_ph: next_states,
                    self._next_legal_actions_mask_ph: next_legal_action_masks,
                    self._gumbel_noise_ph:gumbel1,
                    self._gumbel_noise_2_ph:gumbel2,
                    self._state_mean_ph:[self._state_mean],
                    self._state_std_ph:[self._state_std],
                    self._anchor_ph: anchors, 
                    self._positive_ph: positives, 
                    self._negative_ph: negatives,
                }
            )

           
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
                        self._state_std_ph:[self._state_std],
                        self._gumbel_noise_ph:gumbel1
                    }
                )

                # print("policy loss: ", loss)

                # Update the target networks if updated policies
                self._session.run(self._update_q1)
                self._session.run(self._update_q2)
                self._session.run(self._update_policy)
                self._session.run(self._update_encoder)
        # print("Contrastive Loss: ", contrastive_loss)
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
        
