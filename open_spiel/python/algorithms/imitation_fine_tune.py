# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DQN agent implemented in TensorFlow."""

import collections
import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
from itertools import product
import random
import time
import datetime

from open_spiel.python import rl_agent
from open_spiel.python import simple_nets
from open_spiel.python.utils.replay_buffer import ReplayBuffer
from open_spiel.python.algorithms.psro_v2 import utils
#Config.py for ppo training data
from open_spiel.python.algorithms import config

# Temporarily disable TF2 behavior until code is updated.
tf.disable_v2_behavior()

"""
Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state next_action is_final_step legal_actions_mask rewards_to_go gae")"""

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state is_final_step legal_actions_mask")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9

# Since we are working in joint action space, the output of the Q network will be exponential in the number of players
# We index it so that for a player with player index of i, their action will change every (num_actions) ** i indices in the action output
# For example, if a game has an action space size of 2, for player index 0, their action will be different for every 1 of the outputs
# For player index 1, their action will be different every 2 outputs
# For player index 2, their action will be different every 4 outputs

# To marginalize a player's action, take slices of size (num_actions) ** i every (num_actions) ** (i + 1) indices.
# Then, take the max of them. That represents the marginalized max Q-value of that actions

# To get different marginalized Q-value for each action, start at different points: 0, num_actions ** i, 2 * (num_actions ** i), 3 * (num_actions ** i) etc.

# To calculate the Q value for a particular joint action, just take indices of all of the players and calculate the number in base num_actions


class ImitationFineTune(rl_agent.AbstractAgent):
    """Conservative Q-Learning Agent implementation in TensorFlow.

    See open_spiel/python/examples/breakthrough_dqn.py for an usage example.
    """

    def __init__(self,
                 pre_trained_network,
                 player_id,
                 consensus_kwargs,
                 num_actions, 
                 state_representation_size, 
                 num_players, 
                 turn_based, 
                 prev_policy, 
                 policy_constraint):

        """Initialize the DQN agent."""

        # This call to locals() is used to store every argument used to initialize
        # the class instance, so it can be copied with no hyperparameter change.
        self._kwargs = locals()
        self.session = consensus_kwargs["session"]
        self.num_players = num_players
        self.device = consensus_kwargs["device"]
        self._is_turn_based = turn_based

        self.player_id = player_id
        self.symmetric = consensus_kwargs["symmetric"]
        self._num_actions = num_actions

        self.hidden_layer_size = consensus_kwargs["hidden_layer_size"]
        self.n_hidden_layers = consensus_kwargs["n_hidden_layers"]
        self.rewards_joint = consensus_kwargs["rewards_joint"]
        self.joint_action = consensus_kwargs['joint_action']
        self.discount_factor = consensus_kwargs["discount"]
        # self.entropy_regularization = consensus_kwargs["ppo_entropy_regularization"]
        # self.entropy_regularization_start = self.entropy_regularization

        # BELOW is for R-BVE finetuning
        self.max_buffer_size_fine_tune = consensus_kwargs["max_buffer_size_fine_tune"]
        self.min_buffer_size_fine_tune = consensus_kwargs["min_buffer_size_fine_tune"]
        self.consensus_kwargs = consensus_kwargs
        self.state_representation_size = state_representation_size

        self.layer_sizes = [self.hidden_layer_size] * self.n_hidden_layers

        # Initialize replay
        self._replay_buffer = ReplayBuffer(self.max_buffer_size_fine_tune)
        self._all_trajectories = []
        self._all_action_trajectories = []
        self._all_override_symmetrics = []
        # self._curr_size_batch = 0
        self._seen_observations = set()

        # Initialize the FF network

        self.num_outputs = self._num_actions ** num_players if self.joint_action else self._num_actions

        self.trained_once = False

        # Create placeholders
        self._info_state_ph = tf.placeholder(
            shape=[None, state_representation_size],
            dtype=tf.float32,
            name="info_state_ph")
        self._action_ph = tf.placeholder(
            shape=[None, 1], dtype=tf.int32, name="action_ph")
        self._is_final_step_ph = tf.placeholder(
            shape=[None], dtype=tf.float32, name="is_final_step_ph")
        self._next_info_state_ph = tf.placeholder(
            shape=[None, state_representation_size],
            dtype=tf.float32,
            name="next_info_state_ph")
        self._rewards_ph = tf.placeholder(
            shape=[None], dtype=tf.float32, name="rewards_ph"
        )
        
        self._legal_actions_mask_ph = tf.placeholder(
            shape=[None, num_actions], dtype=tf.float32, name="legal_actions_mask_ph"
            )
        """
        self._entropy_regularization_ph = tf.placeholder(
            shape=(), dtype=tf.float32, name="entropy_regularization"
        )
        self._rewards_to_go_ph = tf.placeholder(
            shape=[None], dtype=tf.float32, name="rewards_to_go_ph"
        )
        self._next_action_ph = tf.placeholder(
            shape=[None, 1], dtype=tf.int32, name="next_action_ph"
        )
        """


        # TODO: Insert a flag whether or not to save the model
        # TODO: Insert path to save model (use the current datetime as the name of the model)
        self._save_model_after_training = consensus_kwargs["save_models"]
        self._load_model_before_training = consensus_kwargs["load_models"]
        self._save_model_path = consensus_kwargs["save_model_path"]
        ####### R-BVE/SARSA Start ########

        self._env_steps = 0
        self._fine_tune_print_counter = 0

        # self._fine_tune_counter = 0

        ##################################################################

        ################### Begin PPO fine-tuning code ###################

        ##################################################################
        """
        self._old_log_probs_ph = tf.placeholder(
            shape=[None, 1],
            dtype=tf.float32,
            name="old_log_probs_ph"
        )
        self._old_values_ph = tf.placeholder(
            shape=[None, 1],
            dtype=tf.float32,
            name="old_values_ph"
        )
        self._gae_ph = tf.placeholder(
            shape=[None, 1],
            dtype=tf.float32,
            name="gae_ph"
        )"""

        self._policy_constraint_weight_ph = tf.placeholder(
            shape=(),
            dtype=tf.float32,
            name="policy_constraint_weight_ph"
        )
        # Trajectories and actions for online fine-tuning without the joint wrapper
        # self.curr_trajectory = []
        # self.action_trajectory = []
        self.policy_constraint_weight = policy_constraint
        self._prev_timestep = None
        self._prev_action = None

        """
        # Create a policy network same size as Q network
        self._policy_network = simple_nets.MLP(state_representation_size,
                                            self.layer_sizes, num_outputs)
        self._policy_network_variables = self._policy_network.variables[:]
        self._exploration_policy_network = simple_nets.MLP(state_representation_size,
                                            self.layer_sizes, num_outputs)
        self._exploration_policy_network_variables = self._exploration_policy_network.variables[:]
        # self._policy_network_copy = simple_nets.MLP(state_representation_size, 
        #                                     self.layer_sizes, num_outputs)
        # self._policy_network_copy_variables = self._policy_network_copy.variables[:]


        # Create a VALUE network same size as Q network
        self._value_network = simple_nets.MLP(state_representation_size, self.layer_sizes, 1)  # self._num_actions)
        self._value_network_variables = self._value_network.variables[:]

        # self._initialize_policy_network = self._create_policy_network(self._policy_network, pre_trained_network)
        self._initialize_exploration_policy_network = self._create_policy_network(self._exploration_policy_network, pre_trained_network)
        # self._save_policy_network = self._create_policy_network(self._policy_network_copy, self._policy_network)
        # self._recover_policy_network = self._create_policy_network(self._policy_network, self._policy_network_copy)
        # self._prev_policy_copy_from = prev_policy

        # Pass observations to policy
        logits = self._policy_network(self._info_state_ph) # [?, num_actions]

        self.probs = tf.nn.softmax(logits, axis=1)  # exps / normalizer # [?, num_actions]

        # Then do tf.log on them to get logprobs
        all_log_probs = tf.math.log(tf.clip_by_value(self.probs, 1e-10, 1.0)) # [?, num_actions]
        self.log_probs = tf.gather(all_log_probs, self._action_ph, axis=1, batch_dims=1) # [?, 1]

        ###################################
        #### Policy constraint section ####
        constrain_logits = self._exploration_policy_network(self._info_state_ph)
        constrain_probs = tf.nn.softmax(constrain_logits, axis=1)

        constrain_all_log_probs = tf.stop_gradient(tf.math.log(tf.clip_by_value(constrain_probs, 1e-10, 1.0)))
        self.constrain_kl_divergence = tf.reduce_mean(tf.reduce_sum(tf.math.multiply(tf.math.exp(constrain_all_log_probs), constrain_all_log_probs - all_log_probs), axis=1))
        ###################################

        ###################################
        # PPO Implementation 
        # Pass observations to value network to get value baseline 
        eps_clip = consensus_kwargs["eps_clip"]
        eps_clip_value = consensus_kwargs["eps_clip_value"]

        self.values = self._value_network(self._info_state_ph) # [?, 1]

        # self.running_returns = []
        # self.average_window = consensus_kwargs["recovery_window"]
        # self.first_returns = None 
        # self.running_return = 0
        # self.curr_returns = 0

        # Subtract from the rewards to go to get advantage values
        print("rtg values check: ", self._rewards_to_go_ph.get_shape(), self.values.get_shape())
        assert (tf.reshape(self._rewards_to_go_ph, [-1, 1])).get_shape() == self.values.get_shape()
        assert self._old_values_ph.get_shape() == self.values.get_shape()
        value_clipped = tf.stop_gradient(self._old_values_ph) + tf.clip_by_value(self.values - tf.stop_gradient(self._old_values_ph), -eps_clip_value, eps_clip_value)

        assert tf.reshape(self._rewards_to_go_ph, [-1, 1]).get_shape() == value_clipped.get_shape()
        value_delta_1 = tf.reshape(self._rewards_to_go_ph, [-1, 1]) - value_clipped
        value_delta_2 = tf.reshape(self._rewards_to_go_ph, [-1, 1]) - self.values  # [?, 1]
        assert value_delta_1.get_shape() == value_delta_2.get_shape()

        advantages = tf.reshape(self._gae_ph, [-1, 1])

        # Calculate entropy
        assert self.probs.get_shape() == all_log_probs.get_shape()
        self.entropy = tf.reduce_mean(-(tf.reduce_sum(self.probs * all_log_probs, axis=1)))

        # L2 regularization 
        # policy_vars = self._policy_network.trainable_variables
        # lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in policy_vars])

        # Calculate actor loss by negative weighting log probs by DETACHED advantage values and adding in entropy regularization with .01 weight
        assert self.log_probs.get_shape() == advantages.get_shape()
        normalized_advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-7)

        assert self.log_probs.get_shape() == tf.reshape(tf.stop_gradient(self._old_log_probs_ph), [-1, 1]).get_shape()
        ratios = tf.math.exp(self.log_probs - tf.reshape(tf.stop_gradient(self._old_log_probs_ph), [-1, 1]))
        assert ratios.get_shape() == normalized_advantages.get_shape()
        surr1 = ratios * tf.stop_gradient(normalized_advantages)
        surr2 = tf.clip_by_value(ratios, 1-eps_clip, 1+eps_clip) * tf.stop_gradient(normalized_advantages)
        self.actor_loss = -tf.reduce_mean(tf.math.minimum(surr1, surr2)) + (self._policy_constraint_weight_ph * self.constrain_kl_divergence) - (self._entropy_regularization_ph * self.entropy) # + lossL2 * self.entropy_regularization #


        # Calculcate critic loss by taking the square of advantages 
        self.critic_loss = tf.reduce_mean(tf.maximum(tf.math.square(value_delta_1), tf.math.square(value_delta_2)))

        # Create separate optimizers for the policy and value network
        self._ppo_policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.consensus_kwargs["fine_tune_policy_lr"])
        self._ppo_value_optimizer = tf.train.AdamOptimizer(learning_rate=self.consensus_kwargs["fine_tune_value_lr"])

        # Learn step
        self._ppo_value_learn_step = self._ppo_value_optimizer.minimize(self.critic_loss)
        self._ppo_policy_learn_step = self._ppo_policy_optimizer.minimize(self.actor_loss)
        """

        self.update_every = consensus_kwargs["sac_update_every"]
        self.batch_size = consensus_kwargs["sac_batch_size"]
        self.sac_value_clip = consensus_kwargs["sac_value_clip"]
        self.tau = .005

        # Initialize policy network, exploration policy, 2 q networks, and 2 target q networks
        
        # Create policy network and exploration policy networks
        self._policy_network = simple_nets.MLP(self.state_representation_size, self.layer_sizes, self.num_outputs)
        self._exploration_policy_network = simple_nets.MLP(self.state_representation_size, self.layer_sizes, self.num_outputs)

        # Create 2 Q-networks and their respective target networks
        self._q1 = simple_nets.MLP(state_representation_size, self.layer_sizes, self._num_actions)
        self._q2 = simple_nets.MLP(state_representation_size, self.layer_sizes, self._num_actions)
        self._q1_target = simple_nets.MLP(state_representation_size, self.layer_sizes, self._num_actions)
        self._q2_target = simple_nets.MLP(state_representation_size, self.layer_sizes, self._num_actions)
        
        # Method to initialize the exploration policy with the offline-trained network
        self._initialize_exploration_policy_network = self._create_policy_network(self._exploration_policy_network, pre_trained_network)

        # Set value for entropy target
        self.entropy_target = -np.log((1.0 / self._num_actions)) * 0.98
        # self.log_alpha = tf.Variable(1.0)

        # self.session.run(self.log_alpha.initializer)
        self.alpha = consensus_kwargs["sac_alpha"] # tf.exp(self.log_alpha)

        # Value network loss

        # Pass in next state into both of the target networks 
        next_q1 = self._q1_target(self._next_info_state_ph) # [?, num_actions]
        next_q2 = self._q2_target(self._next_info_state_ph) # [?, num_actions]

        # Take the min of them element-wise 
        min_next_q = .5 * (next_q1 + next_q2) # tf.math.minimum(next_q1, next_q2) # [?, num_actions]

        # Get the action probabilities for the next state 
        next_action_logits = self._policy_network(self._next_info_state_ph)  # [?, num_actions]

        # Apply legal actions mask so that illegal actions have essentially probability 0 and we do not interfere with softmax nor gradients
        assert next_action_logits.get_shape() == self._legal_actions_mask_ph.get_shape()
        next_action_logits = next_action_logits * (self._legal_actions_mask_ph) + ((1 - self._legal_actions_mask_ph) * ILLEGAL_ACTION_LOGITS_PENALTY)
        next_action_probs = tf.nn.softmax(next_action_logits, axis=1)  # [?, num_actions]

        # Get the log action probabilities (Add a small value to not get NaNs)
        next_action_log_probs = tf.math.log(tf.clip_by_value(next_action_probs, 1e-10, 1.0)) # [?, num_actions]

        # Dot the action probabilities by (min values - self.alpha * log action probabilities)
        assert next_action_probs.get_shape() == min_next_q.get_shape() 
        assert next_action_log_probs.get_shape() == min_next_q.get_shape()
        min_q_target = next_action_probs * (min_next_q - self.alpha * next_action_log_probs)  # [?, num_actions] * [?, num_actions] = [?, num_actions]
            
        # Next Q value is the sum over the axis of actions (an expected Q value)
        min_q_target = tf.reshape(tf.reduce_sum(min_q_target, axis=1), [-1, 1]) # [?, 1]

        # Take away the gradient of the Next Q values
        min_q_target = tf.stop_gradient(min_q_target)  # [?, 1]

        # Get the min_q_target using td target
        assert tf.reshape(self._rewards_ph, [-1, 1]).get_shape() == tf.reshape(self._is_final_step_ph, [-1, 1]).get_shape()
        assert  tf.reshape(self._is_final_step_ph, [-1, 1]).get_shape() == min_q_target.get_shape()
        min_q_target = tf.reshape(self._rewards_ph, [-1, 1]) + (1 - tf.reshape(self._is_final_step_ph, [-1, 1])) * self.discount_factor * min_q_target  # [?, 1]

        # Pass in current state to both value networks
        q1 = self._q1(self._info_state_ph)  # [?, num_actions]
        q2 = self._q2(self._info_state_ph)  # [?, num_actions]

        # Gather the indices of the current action for both 
        q1_gathered = tf.gather(q1, self._action_ph, axis=1, batch_dims=1)  # [?, 1]
        q2_gathered = tf.gather(q2, self._action_ph, axis=1, batch_dims=1)  # [?, 1]

        # Gather the indices of the current action from the target networks 
        q1_gathered_target = tf.stop_gradient(tf.gather(self._q1_target(self._info_state_ph), self._action_ph, axis=1, batch_dims=1)) # [?, 1]
        q2_gathered_target = tf.stop_gradient(tf.gather(self._q2_target(self._info_state_ph), self._action_ph, axis=1, batch_dims=1)) # [?, 1]

        # Calculate the mean squared error of the current Q and the next Q for both of the value networks
        assert q1_gathered.get_shape() == min_q_target.get_shape()
        q1_loss = tf.reduce_mean(tf.math.maximum(tf.math.square(q1_gathered - min_q_target), tf.math.square(q1_gathered_target + tf.clip_by_value(q1_gathered - q1_gathered_target, -self.sac_value_clip, self.sac_value_clip) - min_q_target)))
        q2_loss = tf.reduce_mean(tf.math.maximum(tf.math.square(q2_gathered - min_q_target), tf.math.square(q2_gathered_target + tf.clip_by_value(q2_gathered - q2_gathered_target, -self.sac_value_clip, self.sac_value_clip) - min_q_target)))
        self.critic_loss = q1_loss + q2_loss

        # Policy network loss 
        # Get the values from the VALUE (not target) networks by passing in the current state 
        q1_detached = tf.stop_gradient(q1) # [?, num_actions]
        q2_detached = tf.stop_gradient(q2) # [?, num_actions]

        # Take the element-wise min of the two values
        min_q_detached = tf.math.minimum(q1_detached, q2_detached) # [?, num_actions]

        # Pass in the current state to the policy network  
        action_logits = self._policy_network(self._info_state_ph) # [?, num_actions]
        assert action_logits.get_shape() == self._legal_actions_mask_ph.get_shape()
        action_logits = action_logits * (self._legal_actions_mask_ph) + ((1 - self._legal_actions_mask_ph) * ILLEGAL_ACTION_LOGITS_PENALTY)
        self.action_probs = tf.nn.softmax(action_logits, axis=1)  # [?, num_actions]

        action_log_probs = tf.math.log(tf.clip_by_value(self.action_probs, 1e-10, 1.0)) # [?, num_actions]

        # Calculate the inside term by multiplying (self.alpha * log_probs) - (the min of the two values)
        assert action_log_probs.get_shape() == min_q_detached.get_shape()
        inside_term = (self.alpha * action_log_probs) - min_q_detached  # [?, num_actions]

        # Policy constrain copy and paste
        constrain_logits = self._exploration_policy_network(self._info_state_ph)
        constrain_probs = tf.nn.softmax(constrain_logits, axis=1)  # [?, num_actions]

        constrain_all_log_probs = tf.stop_gradient(tf.math.log(tf.clip_by_value(constrain_probs, 1e-10, 1.0))) # [?, num_actions]
        self.constrain_kl_divergence = tf.reduce_mean(tf.reduce_sum(tf.math.multiply(tf.math.exp(constrain_all_log_probs), constrain_all_log_probs - action_log_probs), axis=1))

        # Multiply the action probabilities by the inside term, sum by the action dimension, and take the mean of the entire thing
        self.policy_loss = tf.reduce_mean(tf.reduce_sum(self.action_probs * inside_term, axis=1)) + (self._policy_constraint_weight_ph * self.constrain_kl_divergence)

        # Alpha loss 
        # TODO: Possible that action_log_probs should be entropy instead?
        # self.alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(action_log_probs + self.entropy_target))

        # Value optimizer and step 
        self._policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.consensus_kwargs["fine_tune_policy_lr"])
        self._value_optimizer = tf.train.AdamOptimizer(learning_rate=self.consensus_kwargs["fine_tune_value_lr"])
        self._alpha_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)

        # Learn step
        self._value_learn_step = self._value_optimizer.minimize(self.critic_loss)
        self._policy_learn_step = self._policy_optimizer.minimize(self.policy_loss)
        # self._alpha_learn_step = self._alpha_optimizer.minimize(self.alpha_loss)

        # Update the target networks 
        self._update_q1_target = self._create_target_network_update_op(self._q1, self._q1_target)
        self._update_q2_target = self._create_target_network_update_op(self._q2, self._q2_target)

        # Savers

        # INSERT WAY TO CHECK IF SAVE_MODEL_PATH is a folder.If not, create recursively 
        pathExists = os.path.exists(self._save_model_path)
        if not pathExists:
            os.makedirs(self._save_model_path)
        files_in_checkpoint_dir = os.listdir(self._save_model_path)
        num_checkpoints = len([f for f in files_in_checkpoint_dir if '.index' in f])

        self._savers = [("policy_network_{}".format(num_checkpoints), tf.train.Saver(self._policy_network.variables))]
                        #("exploration_network_{}".format(num_checkpoints), tf.train.Saver(self._exploration_policy_network.variables)),
                        #("q1_network_{}".format(num_checkpoints), tf.train.Saver(self._q1.variables)),
                        #("q2_network_{}".format(num_checkpoints), tf.train.Saver(self._q2.variables)),
                        #("q1_target_network_{}".format(num_checkpoints), tf.train.Saver(self._q1_target.variables)),
                        #("q2_target_{}".format(num_checkpoints), tf.train.Saver(self._q2_target.variables))]


        self._initialize()

        self.states_seen_in_evaluation = []

    def _create_target_network_update_op(self, q_network, target_q_network):
        """Create TF ops copying the params of the Q-network to the target network.

        Args:
        q_network: A q-network object that implements provides the `variables`
                    property representing the TF variable list.
        target_q_network: A target q-net object that provides the `variables`
                            property representing the TF variable list.

        Returns:
        A `tf.Operation` that updates the variables of the target.
        """
        self._variables = q_network.variables[:]
        self._target_variables = target_q_network.variables[:]
        assert self._variables
        assert len(self._variables) == len(self._target_variables)
        return tf.group([
            tf.assign(target_v, v * self.tau + target_v * (1 - self.tau))
            for (target_v, v) in zip(self._target_variables, self._variables)
        ])

    def clear_state_tracking(self):
        self.states_seen_in_evaluation = []
        return 

    def _create_policy_network(self, policy_network, source_network):
        if source_network != None:
            source_variables = source_network.variables[:]
            policy_variables = policy_network.variables[:]
            assert source_variables
            assert len(source_variables) == len(policy_variables)
            return tf.group([
                tf.assign(target_v, v)
                for (target_v, v) in zip(policy_variables, source_variables)
            ])
        else:
            initialization_network = tf.group(
                *[var.initializer for var in  policy_network.variables[:]]
            )

            return tf.group(*[initialization_network])                                     


    def set_to_fine_tuning_mode(self, is_train_best_response, psro_iteration):

        self.is_train_best_response = is_train_best_response

        # self.recover_policy = is_train_best_response  # We only recover it if it is best response because, otherwise, we would always have a new policy initialized for consensus policies
        # self.epochs = self.consensus_kwargs["epochs_ppo"]
        # self.minibatches = self.consensus_kwargs["minibatches_ppo"]

        self.session.run(self._initialize_exploration_policy_network)  # Initialize the old policy network with the imitation_deep trained network for policy constraint if necessary

        """
        if self._prev_policy_copy_from: # and is_train_best_response:  # Consider initializing it with the previous BR regardless. Might help the pooping out of the policy training
            ref_object = self._prev_policy_copy_from._policy._fine_tune_module 
            ref_policy_network = getattr(ref_object, "_policy_network")


            if self.consensus_kwargs["transfer_policy"]:
                print("Loading previous PPO policy with minimum entropy {}".format(self.consensus_kwargs["transfer_policy_minimum_entropy"]))
                copy_weights = tf.group([
                    tf.assign(vb, va)
                    for (va, vb) in zip(ref_policy_network.variables, self._policy_network.variables)
                ])
                self.session.run(copy_weights)
                # self.session.run(self._save_policy_network)  # Save the network WITHOUT noise. That way, we don't compound noise over PSRO iterations

                # TODO: Consider adding noise here. Lots of the issues with "pooping out" has to do with the starting point it seems. Getting a variety of starting points might help to get it out of there
                copy_weights = tf.group([
                    tf.assign(vb, va * (1 + .05 * tf.random.normal(va.shape)))
                    for (va, vb) in zip(ref_policy_network.variables, self._policy_network.variables)
                ])
                self.session.run(copy_weights)
                
        # else:
        #     print("Loading policies from offline learning: ")
        #     self.session.run(self._initialize_policy_network)  # Initialize the policy with the imitation_deep trained network. If not consensus_imitation, the network initialization is random
        #     self.session.run(self._save_policy_network)
        """

        if self._load_model_before_training and psro_iteration < self.consensus_kwargs["num_iterations_load_only"]:
            
            print("")
            print("Attempting to load model for psro iteration ", psro_iteration)
            self.restore(psro_iteration)
            print("Successfully restored checkpoint for psro iteration ", psro_iteration, "\n")

        self.policy_constraint_weight = 0 if self.is_train_best_response else self.policy_constraint_weight
        
    def reset_buffers(self):
        self._replay_buffer.reset()

    def step(self, time_step, is_evaluation=False, add_transition_record=True):
        """Returns the action to be taken and updates the Q-network if needed.

        Args:
          time_step: an instance of rl_environment.TimeStep.
          is_evaluation: bool, whether this is a training or evaluation call.
          add_transition_record: Whether to add to the replay buffer on this step.

        Returns:
          A `rl_agent.StepOutput` containing the action probs and chosen action.
        """
        # Act step: don't act at terminal info states or if its not our turn.
        # print("LENGTH OF BUFFER: ", len(self._replay_buffer))
        if (not time_step.last()) and (
                time_step.is_simultaneous_move() or
                self.player_id == time_step.current_player() or self.symmetric):

            # This is a weird issue with the current code framework
            if self.symmetric:
                # If symmetric, then having a NOT simultaneous move implies that it is updating the empirical game. Time_step.current_player is correctly set corresponding to the player
                # However, if it is a simultaneous move, then we are working with BR. self.player_id is set manually from rl_oracle.py's sample_episode to make sure we get the right observation
                player = (time_step.current_player() if not time_step.is_simultaneous_move() else self.player_id)
            else:
                # If it's not symmetric, then each agent is given one policy corresponding to player_id
                player = self.player_id

            info_state = time_step.observations["info_state"][player]

            # with tf.device(self.device):
            info_state = np.reshape(info_state, [1, -1])


            if self.joint_action:
                raise NotImplementedError

            else:
                # probs = np.zeros(self._num_actions)
                legal_actions = time_step.observations["legal_actions"][player]

                legal_actions_mask = np.zeros(self._num_actions)
                legal_actions_mask[legal_actions] = 1.0
                
                probs = self.session.run(self.action_probs, feed_dict={self._info_state_ph:info_state, self._legal_actions_mask_ph:[legal_actions_mask]})[0]
                legal_probs = probs[legal_actions]
                legal_probs = legal_probs / np.sum(legal_probs)
                action = utils.random_choice(legal_actions, legal_probs)

                probs[action] = 1.0
            # TODO: Remove this. THis is for experiment
            # if is_evaluation:
            #     self.states_seen_in_evaluation.append(''.join(map(str, time_step.observations["info_state"][self.player_id])))
        else:
            action = None
            probs = []

        if not is_evaluation:
            # if time_step.rewards != None:
            #     self.running_return += time_step.rewards[0] if self.symmetric else time_step.rewards[self.player_id]

            # self.curr_trajectory.append(time_step)
            # if action != None:
            #     self.action_trajectory.append([action])

            if action != None:  # action could be 0
                self._env_steps += 1
            
            if add_transition_record and self._prev_timestep:
                # If it is simultaneous, then pass in override symmetric True. This will indicate that we only want transitions from one of the agents (the one that is creating new strategy)
                if not self._is_turn_based:
                    self.add_transition(self._prev_timestep, [self._prev_action], time_step, override_symmetric=True)
                # If it is turn-based, then pass in override player [self.player_id]. This will indicate that we only want transitions from the agent corresponding to this policy.
                else:
                    # We note that in symmetric implementation, during BR calculation, player will always be 0 and self.player_id will also always be 0
                    self.add_transition(self._prev_timestep, [self._prev_action], time_step, override_player=[self.player_id])
            
            if len(self._replay_buffer) > self.min_buffer_size_fine_tune and self._env_steps % self.update_every == 0:
                self.fine_tune()

            if time_step.last():
                # if add_transition_record:
                    

                    # self.add_trajectory(self.curr_trajectory, self.action_trajectory, override_symmetric=True)  # we want to override symmetric because we are now training individually against other targets that are not ourselves
                # self.curr_trajectory = []
                # self.action_trajectory = []
                # self.insert_transitions()


                # self._curr_size_batch = 0

                # self.running_returns.append(self.running_return)
                # self.running_return = 0

                """
                if len(self._replay_buffer) > self.min_buffer_size_fine_tune:
                    # TODO: Change this conditional so that we check the size of the buffer
                    self.fine_tune()
                """

                self._prev_timestep = None
                self._prev_action = None

            else:
                self._prev_timestep = time_step
                self._prev_action = action
                        
                """
                if len(self.running_returns) > self.average_window:
                    if self.first_returns == None:
                        self.first_returns = sum(self.running_returns) / len(self.running_returns)
                """
        return rl_agent.StepOutput(action=action, probs=probs)

    def _full_checkpoint_name(self, checkpoint_dir, name):
        return os.path.join(checkpoint_dir, name)
    
    def _latest_checkpoint_filename(self, name):
        return name + "_latest"

    def save(self, checkpoint_dir):
        for name, saver in self._savers:
            print("Saving model with full name: ", self._full_checkpoint_name(checkpoint_dir, name))
            path = saver.save(
                self.session,
                self._full_checkpoint_name(checkpoint_dir, name)
            )
    
    def restore(self, index):
        for name, saver in self._savers:
            name_without_last_index = "_".join(name.split('_')[:-1]) + "_"
            full_checkpoint_dir = self._full_checkpoint_name(self._save_model_path, name_without_last_index + str(index))
            saver.restore(self.session, full_checkpoint_dir)
            
    
    def post_training(self):
        # Post training, if we somehow find a policy worse than the one we initialized with, recover the previous policy
        """
        self.running_returns = self.running_returns[-self.average_window * 5:]
        long_term_value = sum(self.running_returns) / len(self.running_returns)
        short_term_value = sum(self.running_returns[-self.average_window:]) / self.average_window

        
        if long_term_value < self.first_returns or short_term_value < self.first_returns:#  and self.recover_policy:
            print("Recovering previous policy with expected return of {}. Long term value was {} and short term was {}.".format(self.first_returns, long_term_value, short_term_value))
            self.session.run(self._recover_policy_network)
        """

        if self._save_model_after_training:
            checkpoint_dir = self._save_model_path
            print("Saving model...")
            self.save(checkpoint_dir)
            print("Successfully saved model at checkpoint directory: ", checkpoint_dir, "\n")
        return

    def fine_tune(self):
                
        # self._env_steps += len(self._replay_buffer)
        # self.entropy_regularization =  max((1 - self._env_steps / (self.consensus_kwargs["entropy_decay_duration"] * self.consensus_kwargs["steps_fine_tune"])), 0) * self.entropy_regularization_start
        # self._fine_tune_counter += 1
        self._fine_tune_print_counter -= 1

        # epochs = self.epochs
        # minibatches = self.minibatches

        
        transitions = self._replay_buffer.sample(self.batch_size)
        # Run value network update session
        info_states = [t.info_state for t in transitions]
        actions = [[t.action] for t in transitions]
        next_info_states = [t.next_info_state for t in transitions]
        rewards = [t.reward for t in transitions]
        is_final_step = [t.is_final_step for t in transitions] 
        legal_actions_mask = [t.legal_actions_mask for t in transitions]

        value_loss, _ = self.session.run(
            [self.critic_loss, self._value_learn_step],
            feed_dict={
                self._info_state_ph: info_states,
                self._action_ph: actions,
                self._rewards_ph: rewards,
                self._next_info_state_ph: next_info_states,
                self._is_final_step_ph: is_final_step,
                self._legal_actions_mask_ph: legal_actions_mask,
                self._policy_constraint_weight_ph: self.policy_constraint_weight,
            })

        # Run policy network update session 
        policy_loss, _ = self.session.run(
            [self.policy_loss, self._policy_learn_step],
            feed_dict={
                self._info_state_ph: info_states,
                self._action_ph: actions,
                self._rewards_ph: rewards,
                self._next_info_state_ph: next_info_states,
                self._is_final_step_ph: is_final_step,
                self._legal_actions_mask_ph: legal_actions_mask,
                self._policy_constraint_weight_ph: self.policy_constraint_weight,
            })

        # Update temperature
        """
        alpha_loss, _ = self.session.run(
            [self.alpha_loss, self._alpha_learn_step],
            feed_dict={
                self._info_state_ph: info_states,
                self._action_ph: actions,
                self._rewards_ph: rewards,
                self._next_info_state_ph: next_info_states,
                self._is_final_step_ph: is_final_step,
                self._legal_actions_mask_ph: legal_actions_mask,
                self._policy_constraint_weight_ph: self.policy_constraint_weight,
            }) """
            
        # Run target network update
        self.session.run(self._update_q1_target)
        self.session.run(self._update_q2_target)
            
            
        # transitions = self._replay_buffer.sample(len(self._replay_buffer))

        """
        info_states = [t.info_state for t in transitions]
        actions = [[t.action] for t in transitions]

        
        old_log_probs, old_values = self.session.run(
            [self.log_probs, self.values],
            feed_dict={
                self._info_state_ph: info_states,
                self._action_ph: actions
            }
        )
        """

        """
        indices = np.arange(len(transitions))
        size_minibatch = len(transitions) // minibatches
        # For number of minibatches
        for _ in range(epochs):
            # Get some random order of all of the transitions
            np.random.shuffle(indices)


            for i in range(minibatches):
                start = i*size_minibatch
                end = (i+1)*size_minibatch if i < minibatches - 1 else len(indices)

                subset = indices[start:end]
                info_states = [transitions[j].info_state for j in subset]
                actions = [[transitions[j].action] for j in subset]
                next_actions =[[transitions[j].next_action] for j in subset]
                next_info_states = [transitions[j].next_info_state for j in subset]
                rewards_to_go = [transitions[j].rewards_to_go for j in subset]
                gae = [[transitions[j].gae] for j in subset]
                old_log_probs_subset = [old_log_probs[j] for j in subset]

                actor_loss, _, entropy, probs, log_probs, kl = self.session.run(
                [self.actor_loss, self._ppo_policy_learn_step, self.entropy, self.probs, self.log_probs, self.constrain_kl_divergence],
                feed_dict={
                    self._info_state_ph: info_states,
                    self._action_ph: actions,
                    self._rewards_to_go_ph: rewards_to_go,
                    self._old_log_probs_ph: old_log_probs_subset,
                    self._gae_ph: gae,
                    self._policy_constraint_weight_ph: self.policy_constraint_weight,
                    # self._entropy_regularization_ph: self.entropy_regularization
                })
            # config.actor_loss_list.append(actor_loss)
            config.entropy_list.append(entropy)
            config.kl_list.append(kl)

        # For number of minibatches
        for _ in range(epochs):
            np.random.shuffle(indices)


            for i in range(minibatches):
                start = i*size_minibatch
                end = (i+1)*size_minibatch if i < minibatches - 1 else len(indices)

                subset = indices[start:end]
                info_states = [transitions[j].info_state for j in subset]
                rewards_to_go = [transitions[j].rewards_to_go for j in subset]
                old_values_subset = [old_values[j] for j in subset]

                value_loss, _ = self.session.run(
                    [self.critic_loss, self._ppo_value_learn_step],
                    feed_dict={
                        self._info_state_ph: info_states,
                        self._rewards_to_go_ph: rewards_to_go,
                        self._old_values_ph: old_values_subset
                    })
            # config.value_loss_list.append(value_loss)
        """
    
        if (self._fine_tune_print_counter <= 0):
            print("SAC env steps so far ", self._env_steps, self.policy_constraint_weight) # sum(config.entropy_list) / len(config.entropy_list), sum(config.kl_list) / len(config.kl_list),
            # print("Reward scaling mean, std: ", self.reward_scaler.rs.mean, self.reward_scaler.rs.std)
            self._fine_tune_print_counter = 20000
        
        # self._replay_buffer.reset()

        return


    def _initialize(self):
        initialization_policy = tf.group(
            *[var.initializer for var in self._policy_network.variables[:]]
        )
        initialization_q1 = tf.group(
            *[var.initializer for var in self._q1.variables[:]]
        )
        initialization_q2 = tf.group(
            *[var.initializer for var in self._q2.variables[:]]
        )
        initialization_q1_target = tf.group(
            *[var.initializer for var in self._q1_target.variables[:]]
        )
        initialization_q2_target = tf.group(
            *[var.initializer for var in self._q2_target.variables[:]]
        )
        initialization_value_opt = tf.group(
            *[var.initializer for var in self._value_optimizer.variables()]
        )
        initialization_policy_opt = tf.group(
            *[var.initializer for var in self._policy_optimizer.variables()]
        )
        initialization_alpha_opt = tf.group(
            *[var.initializer for var in self._alpha_optimizer.variables()]
        )
        """
        initialization_policy_copy = tf.group(
            *[var.initializer for var in self._policy_network_copy_variables]
        )"""

        self.session.run(
            tf.group(*[
                initialization_policy,
                initialization_q1,
                initialization_q2,
                initialization_q1_target,
                initialization_q2_target,
                initialization_value_opt,
                initialization_policy_opt,
                initialization_alpha_opt
            ]))

    def cumsum(self, x, discount):
        vals = [None for _ in range(len(x))]
        curr = 0
        for i in range(len(x) - 1, -1, -1):
            vals[i] = curr = x[i] + discount  * curr
        return vals

    """ Adds the trajectory consisting only of transitions relevant to the current player (only self.player_id if turn-based or all of them if simultaneous)"""
    """
    def add_trajectory(self, trajectory, action_trajectory, override_symmetric=False):
        self._curr_size_batch += len(action_trajectory)
        self._all_trajectories.append(trajectory)
        self._all_action_trajectories.append(action_trajectory)
        self._all_override_symmetrics.append(override_symmetric)
    """

    """
    def insert_transitions(self):
        all_gae = []
        all_rewards_to_go = []

        for trajectory in self._all_trajectories:
            reward_arrays = [np.array(timestep.rewards) for timestep in trajectory[1:]]  # the first timestep has None value
            rewards_to_go = [np.zeros(self.num_players) for _ in range(len(trajectory) - 1)]

            # For each of the trajectories and action trajectories, calculate the rewards to go
            rewards_to_go = self.cumsum(reward_arrays, self.discount_factor)
            all_rewards_to_go.append(rewards_to_go)

        for i, trajectory in enumerate(self._all_trajectories):
            gae = [np.zeros(self.num_players) for _ in range(len(trajectory) - 1)]
            values = [np.zeros(self.num_players) for _ in range(len(trajectory))]



            for p in range(self.num_players):
                observations = [timestep.observations["info_state"][p] for timestep in trajectory]
                vals = self.session.run([self.values], feed_dict={self._info_state_ph:observations})[0]
                vals = np.array(vals)

                for j, val in enumerate(values):
                    val[p] = vals[j] 
            
            reward_arrays = [np.array(timestep.rewards) for timestep in trajectory[1:]]
            deltas = [reward_arrays[j] + self.discount_factor * values[j+1] - values[j] for j in range(len(reward_arrays))]
            gae = self.cumsum(deltas, self.discount_factor * .95)

            # Append gae and normalized rewards to go
            all_gae.append(gae)


        for curr, (trajectory, action_trajectory) in enumerate(zip(self._all_trajectories, self._all_action_trajectories)):
            rewards_to_go = all_rewards_to_go[curr]
            gae = all_gae[curr]

            for i in range(len(trajectory) - 1):
                if not self._is_turn_based:
                    # NOTE: If is_symmetric, then add_transition will add observations/actions from BOTH players already
                    # NOTE: Also insert action_trajectory[i+1]. If it is the last i, then we let action be 0 because it won't be used anyway
                    next_action = action_trajectory[i+1] if (i+1) < len(action_trajectory) else [0]
                    self.add_transition(trajectory[i], action_trajectory[i], trajectory[i+1], next_action, ret=rewards_to_go[i], gae=gae[i], override_symmetric=self._all_override_symmetrics[curr])
                else:
                    # NOTE: Assume that anything called using add_trajectory already filters out for the relevant transitions 
                    player = trajectory[i].observations["current_player"]

                    next_action = action_trajectory[i+1] if (i+1) < len(action_trajectory) else [0] 
                    self.add_transition(trajectory[i], action_trajectory[i], trajectory[i+1], next_action, ret=rewards_to_go[i], gae=gae[i], override_player=[player])
                    
        self._all_trajectories = []
        self._all_action_trajectories = []
        self._all_override_symmetrics = []
    """

    def add_transition(self, prev_time_step, prev_action, time_step, override_symmetric=False, override_player=[]):
        """Adds the new transition using `time_step` to the replay buffer.

        Adds the transition from `self._prev_timestep` to `time_step` by
        `self._prev_action`.

        Args:
          prev_time_step: prev ts, an instance of rl_environment.TimeStep.
          prev_action: list of int, action taken at `prev_time_step`.
          time_step: current ts, an instance of rl_environment.TimeStep.
        """
        player_list = [i for i in range(self.num_players)] if self.symmetric and not override_symmetric else [self.player_id]
        if len(override_player) > 0:
            player_list = override_player

        if self.joint_action:
            raise NotImplementedError
        else:
            for p in player_list:
                o = prev_time_step.observations["info_state"][p][:]

                assert not self.rewards_joint  # because gae is not valid

                r = sum(time_step.rewards) if self.rewards_joint else time_step.rewards[p] # WELFARE
                # rewards_to_go = sum(ret) if self.rewards_joint else ret[p]

                # NOTE: We want to keep all the transitions consistent...as if it was from one player's perspective.
                # So, we need to switch the actions for consistency (this implementation only works for 2-player games)
                # If we didn't switch, then the same action for the same observation can yield different results depending on which player we took that transition from

                # Since the step() function assumes by symmetry that observations come from player0, we need to make sure that all
                # transitions are from player0's perspective, meaning the action applied to the observed player's observation must come first

                # NOTE: Actions are always one-dimensional because these transitions are created directly from step()
                assert len(prev_action) == 1 # and len(action) == 1
                store_action = prev_action[0]
                # store_next_action = action[0]

                legal_actions = (time_step.observations["legal_actions"][p])
                legal_actions_mask = np.zeros(self._num_actions)
                legal_actions_mask[legal_actions] = 1.0

                transition = Transition(
                    info_state=(
                        prev_time_step.observations["info_state"][p][:]),
                    action=store_action,
                    reward=r,
                    next_info_state=time_step.observations["info_state"][p][:],
                    is_final_step=float(time_step.last()),
                    legal_actions_mask=legal_actions_mask,
                )

                self._replay_buffer.add(transition)

    def get_weights(self):
        return [0]

    def copy_with_noise(self, sigma=0.0, copy_weights=False):
        """Copies the object and perturbates it with noise.

        Args:
          sigma: gaussian dropout variance term : Multiplicative noise following
            (1+sigma*epsilon), epsilon standard gaussian variable, multiplies each
            model weight. sigma=0 means no perturbation.
          copy_weights: Boolean determining whether to copy model weights (True) or
            just model hyperparameters.

        Returns:
          Perturbated copy of the model.
        """
        _ = self._kwargs.pop("self", None)
        copied_object = Imitation(**self._kwargs)

        return copied_object
