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

from open_spiel.python import rl_agent
from open_spiel.python import simple_nets
from open_spiel.python.utils.replay_buffer import ReplayBuffer
from open_spiel.python.algorithms.psro_v2 import utils
from open_spiel.python.algorithms.imitation_fine_tune import ImitationFineTune

# Temporarily disable TF2 behavior until code is updated.
tf.disable_v2_behavior()

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state next_action is_final_step legal_actions_mask rewards_to_go")

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

class Imitation(rl_agent.AbstractAgent):
    """Conservative Q-Learning Agent implementation in TensorFlow.

    See open_spiel/python/examples/breakthrough_dqn.py for an usage example.
    """

    def __init__(self,
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


        self.training_steps = consensus_kwargs["training_steps"]
        self.minimum_entropy = consensus_kwargs["minimum_entropy"]
        self.lr = consensus_kwargs["deep_network_lr"]
        self.batch = consensus_kwargs["batch_size"]
        self.hidden_layer_size = consensus_kwargs["hidden_layer_size"]
        self.n_hidden_layers = consensus_kwargs["n_hidden_layers"]
        self.rewards_joint = consensus_kwargs["rewards_joint"]
        self.alpha = consensus_kwargs["alpha"]
        self.joint_action = consensus_kwargs['joint_action']
        self.boltzmann = consensus_kwargs["boltzmann"]
        self.update_target_network_every = consensus_kwargs["update_target_every"]
        self.tau = consensus_kwargs["tau"]
        self.discount_factor = consensus_kwargs["discount"]

        # BELOW is for R-BVE
        self.eta = consensus_kwargs["eta"]# .05 
        self.beta = consensus_kwargs["beta"]# .5

        # BELOW is for R-BVE finetuning 
        self.max_buffer_size_fine_tune = consensus_kwargs["max_buffer_size_fine_tune"]
        self.min_buffer_size_fine_tune = consensus_kwargs["min_buffer_size_fine_tune"]

        self.actor_loss_list = []
        self.value_loss_list = []
        self.entropy_list = []

        # Look at above note about marginalizing Q-values for the joint action
        self.player_marginal_indices = [[[] for _ in range(num_actions)] for _ in range(num_players)]  # action index -> list of indices for marginalization
        self.joint_index_to_joint_actions ={}  # maps joint_action_index -> all players' marginalized actions
        max_num_actions = self._num_actions ** num_players

        all_actions = list(product(list(range(num_actions)), repeat=num_players))
        # print("THERE ARE A TOTAL OF {} ACTIONS IN JOINT SPACE".format(len(all_actions)))
        # print("Actions here: {}".format(all_actions))

        for joint_action in all_actions:
            joint_action_index = self._calculate_joint_action_index(np.array(joint_action).reshape(1, -1))
            for p in range(num_players):
                marginalized_action = joint_action[p]  # in symmetric case, self.player_id will ALWAYS be 0
                self.player_marginal_indices[p][marginalized_action].append(int(joint_action_index[0][0]))
            self.joint_index_to_joint_actions[joint_action_index[0][0]] = joint_action
        
        self.player_marginal_indices = np.array(self.player_marginal_indices, dtype=np.dtype(int))
        
        self.layer_sizes = [self.hidden_layer_size] * self.n_hidden_layers

        # Initialize replay
        self._replay_buffer = ReplayBuffer(np.inf)
        self._seen_observations = set()

        # Initialize the FF network 

        num_outputs = self._num_actions ** num_players if self.joint_action else self._num_actions 
        self._q_network = simple_nets.MLP(state_representation_size, 
                                    self.layer_sizes, num_outputs)

        self._target_q_network = simple_nets.MLP(state_representation_size,
                                             self.layer_sizes, num_outputs)

        self._output_policy = simple_nets.MLP(state_representation_size, 
                                    self.layer_sizes, num_outputs)

        # self._policy_variables = self.policy_net.variables[:] 
        self._q_variables = self._q_network.variables[:]
        self._target_q_variables = self._target_q_network.variables[:]
        self.trained_once = False
        
        # Create placeholders
        self._info_state_ph = tf.placeholder(
            shape=[None, state_representation_size],
            dtype=tf.float32,
            name="info_state_ph")
        self._action_ph = tf.placeholder(
            shape=[None, 1], dtype=tf.int32, name="action_ph")
        self._reward_ph = tf.placeholder(
            shape=[None], dtype=tf.float32, name="reward_ph")
        self._is_final_step_ph = tf.placeholder(
            shape=[None], dtype=tf.float32, name="is_final_step_ph")
        self._next_info_state_ph = tf.placeholder(
            shape=[None, state_representation_size],
            dtype=tf.float32,
            name="next_info_state_ph")
        self._rewards_to_go_ph = tf.placeholder(
            shape=[None], dtype=tf.float32, name="rewards_to_go_ph"
        )
        self._next_action_ph = tf.placeholder(
            shape=[None, 1], dtype=tf.int32, name="next_action_ph"
        )

        # NOTE: ONly supported for joint_action=False
        self._legal_actions_mask_ph = tf.placeholder(
            shape=[None, num_outputs],
            dtype=tf.float32,
            name="legal_actions_mask_ph")

        # Define placeholder functions and operations

        illegal_actions = 1 - self._legal_actions_mask_ph
        illegal_logits = illegal_actions * ILLEGAL_ACTION_LOGITS_PENALTY

        # Q values for current states
        self._q_values = self._q_network(self._info_state_ph)

        # Defining Q value targets for MSE loss
        self._target_q_values = self._target_q_network(self._next_info_state_ph)


        ####### CQL/Double-CQL Start #######        
        # next_q_double = self._q_network(self._next_info_state_ph)
        # max_next_a = tf.math.argmax((tf.math.add(tf.stop_gradient(next_q_double), illegal_logits)), axis=-1)
        # max_next_q = tf.stop_gradient(tf.gather(self._target_q_values, max_next_a, axis=1, batch_dims=1))

        # max_next_q = tf.reduce_max(tf.stop_gradient(self._target_q_values), axis=-1)

        self._update_target_network = self._create_target_network_soft_update_op(
                                        self._q_network, self._target_q_network)

        # TODO: Modify the target to just be the return value place holders without the discount factor at all
        # target = (self._reward_ph + (1 - self._is_final_step_ph) * self.discount_factor * max_next_q)

        ####### CQL/Double-CQL End #######

        ####### R-BVE/SARSA Start ########
        self._fine_tune_mode_ph = tf.placeholder(
            shape=(),
            dtype=tf.bool,
            name="fine_tune_mode_ph")
        
        self._fine_tune_steps = 0
        self._fine_tune_learn_steps = 0
        
        next_a_q =  tf.gather(tf.stop_gradient(self._target_q_values), self._next_action_ph, axis=1, batch_dims=1)

        # Double Q for fine-tuning
        next_q_double = self._q_network(self._next_info_state_ph)
        max_next_a = tf.math.argmax((tf.math.add(tf.stop_gradient(next_q_double), illegal_logits)), axis=-1)
        max_next_q = tf.stop_gradient(tf.gather(self._target_q_values, max_next_a, axis=1, batch_dims=1))

        next_a_q = tf.cond(self._fine_tune_mode_ph, 
                            lambda: max_next_q, 
                            lambda: next_a_q)

        target = tf.reshape(self._reward_ph, [-1, 1]) + (1 - tf.reshape(self._is_final_step_ph, [-1, 1])) * self.discount_factor * tf.reshape(next_a_q, [-1, 1])
        ######## R-BVE/SARSA End #########

        target = tf.reshape(target, [-1, 1])

        ##### CQL Regularization Start #####
        """
        minimize_q_function = tf.math.reduce_logsumexp(self._q_values, axis=1) 

        maximize_q_function = tf.gather(self._q_values, self._action_ph, axis=1, batch_dims=1)
        """
        ##### CQL Regularization End #######


        # True predictions for the given joint actions
        predictions = tf.gather(self._q_values, self._action_ph, axis=1, batch_dims=1)# action_indices)

        
        ##### R-BVE Regularization Start ######
        print("Start: ", self._q_values.get_shape(), predictions.get_shape())
        minimize_q_function = self._q_values - predictions + self.eta  # [?, num_actions] - [?, 1] + scalar
        
        # Create a mask of ones 
        one_hots_each_action = tf.one_hot(tf.reshape(self._action_ph, [-1]), depth=num_outputs)
        print("one hots: ", one_hots_each_action.get_shape())
        mask = tf.math.abs(one_hots_each_action - 1.0)  # equivalent to binary NOT operation
        print("mask and minimize: ", minimize_q_function.get_shape(), mask.get_shape())
        minimize_q_function = minimize_q_function * mask
        minimize_q_function = tf.clip_by_value(minimize_q_function, clip_value_min=0, clip_value_max=np.inf)
        minimize_q_function = tf.reduce_sum(tf.math.square(minimize_q_function), axis=1)

        estimate_mean_value = tf.reduce_mean(self._rewards_to_go_ph)
        weights = tf.math.exp((self._rewards_to_go_ph - estimate_mean_value) / self.beta)
        print("Mulitply weights: ", minimize_q_function.get_shape(), weights.get_shape())
        assert minimize_q_function.get_shape() == weights.get_shape()
        minimize_q_function = weights * minimize_q_function

        maximize_q_function = 0

        # Alpha will act as lambda in R-BVE regularization
        ##### R-BVE Regularization End #####
        
        # Calculate CQL Q-network loss
        loss_class = tf.losses.mean_squared_error
        self._bellman_loss = tf.reduce_mean(loss_class(labels=target, predictions=predictions))

        # NOTE: CQL multiples self._bellman_loss by .5
        self._q_loss = self._bellman_loss + tf.cond(self._fine_tune_mode_ph, lambda: 0.0, lambda: (self.alpha * tf.reduce_mean(minimize_q_function - maximize_q_function)))

        self._q_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        self._q_learn_step = self._q_optimizer.minimize(self._q_loss)

        self.running_not_seen_steps = 0

        self._fine_tune_mode = False 

        # Change step to query the policy network instead of the q network
        # Change initialize method to include the new networks and the new optimizers

        self._fine_tune_module = ImitationFineTune(self._output_policy,
                 player_id,
                 consensus_kwargs,
                 num_actions, 
                 state_representation_size, 
                 num_players,
                 self._is_turn_based,
                 prev_policy, 
                 policy_constraint)
        
        # This is for policy extraction 
        log_probs = self._output_policy(self._info_state_ph)
        loss_class = tf.losses.softmax_cross_entropy 

        # Convert the actions to one hot vectors 
        one_hot_vectors = tf.one_hot(tf.reshape(self._action_ph, [-1]), depth=num_outputs)

        # Plug into cross entropy class 
        self._policy_extraction_loss = tf.reduce_mean(loss_class(one_hot_vectors, log_probs))

        self._policy_extraction_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self._policy_extraction_step = self._policy_extraction_optimizer.minimize(self._policy_extraction_loss)

        self._initialize()

        self.states_seen_in_evaluation = []


    def clear_state_tracking(self):
        self.states_seen_in_evaluation = []
        return 

    def _create_target_network_soft_update_op(self, q_network, target_q_network):
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
            tf.assign(target_v, v)#  self.tau * v + (1 - self.tau) * target_v) #
            for (target_v, v) in zip(self._target_variables, self._variables)
        ])

    def _calculate_joint_action_index(self, joint_action):
        # Joint_action batch. Return the action index for each 
        # print(joint_action)
        indices = np.zeros((joint_action.shape[0], 1), dtype=np.int32)
        for i in range(joint_action.shape[1]):
            a = joint_action[:, i:i+1]
            diff = a * (self._num_actions ** i)
            indices = indices + diff
        return indices

    def set_to_fine_tuning_mode(self, is_train_best_response):
        if not is_train_best_response: # Only do this if we had pretrained the policy on something else and we want to transfer this to PPO 
            print("Extracting policy from q function: ")

            length = len(self._replay_buffer)
            dataset = self._replay_buffer.sample(length)  # samples without replacement so take the entire thing. Random order
            indices = list(range(length))

            all_info_states = [t.info_state for t in dataset]
            print("All info states: ", len(all_info_states))

            all_q_values = self.session.run([self._q_values], feed_dict={self._info_state_ph: all_info_states})[0]
            all_action_labels = np.reshape(np.argmax(all_q_values, axis=1), [-1, 1])

            epoch = 0
            average_loss = np.inf
            while epoch < (50) and (not average_loss < .1): # if not self._fine_tuning_mode else self.training_steps / 5):
                epoch += 1
                i, batch, loss_total = 0, 0, 0
                randomized_indices = np.arange(length)
                np.random.shuffle(randomized_indices)
                while i < length:
                    chosen_indices = randomized_indices[i: min(length, i+self.batch)] 
                    
                    curr_info_states = [all_info_states[i] for i in chosen_indices]
                    curr_actions = [all_action_labels[i] for i in chosen_indices]

                    start = time.time()
                    loss, _= self.session.run(
                        [self._policy_extraction_loss, self._policy_extraction_step],
                        feed_dict={
                            self._info_state_ph: curr_info_states,
                            self._action_ph: curr_actions, 
                        })
                    # print("R-BVE learn time: ", time.time() - start)
                    # print("Example check: ", obs_to_actions[''.join(map(str, info_states[0]))], q_values[0])

                    i += self.batch
                    batch += 1
                    loss_total += loss

                average_loss = loss_total / float(batch)
                print("Average loss in policy extraction: ", average_loss)

        self._replay_buffer.reset()
        self._fine_tune_module.set_to_fine_tuning_mode(is_train_best_response)
        self._fine_tune_mode = True
    
    def get_epsilon_fine_tune(self):
        return max(.01, .2 - ((.2 - .01) * self._fine_tune_steps / 500000))


    def step(self, time_step, is_evaluation=False, add_transition_record=True):
        """Returns the action to be taken and updates the Q-network if needed.

        Args:
          time_step: an instance of rl_environment.TimeStep.
          is_evaluation: bool, whether this is a training or evaluation call.
          add_transition_record: Whether to add to the replay buffer on this step.

        Returns:
          A `rl_agent.StepOutput` containing the action probs and chosen action.
        """
        if self._fine_tune_mode:
            # Edge case: set the player_id for the fine tune mode to match this module 
            self._fine_tune_module.player_id = self.player_id
            return self._fine_tune_module.step(time_step, is_evaluation, add_transition_record)

        # Act step: don't act at terminal info states or if its not our turn.
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
                """
                # with tf.device(self.device):
                if len(time_step.observations["global_state"]) > 0:
                    info_state =  np.reshape(time_step.observations["global_state"][0], [1, -1])
                else:
                    info_state = np.reshape(np.concatenate(time_step.observations["info_state"]), [1, -1])
                
                start = time.time()
                joint_q_values = self.session.run(self._q_values, feed_dict={self._info_state_ph: info_state})[0]
                # print("R-BVE network action: ", time.time() - start)

                all_legal_actions = time_step.observations["legal_actions"]
                for i in range(self._num_actions ** self.num_players):
                    joint_action = self.joint_index_to_joint_actions[i]
                    individual_legal_actions = [joint_action[j] in all_legal_actions[j] for j in range(self.num_players)]
                    joint_q_values[i] =  ILLEGAL_ACTION_LOGITS_PENALTY if not all(individual_legal_actions) else joint_q_values[i]
                
                # NOTE: Even if the game is symmetric, since we are working in joint space, we need player specific marginal indices. Otherwise, we will always taking the joint q values and mapping them according to player0's actions, not any player.
                # actions_mapped_to_values = np.take(joint_q_values, self.player_marginal_indices[player])

                # marginalized_q_values = np.max(actions_mapped_to_values, axis=1)

                probs = np.zeros(self._num_actions)

                # legal_actions = time_step.observations["legal_actions"][player]

                # legal_q_values = marginalized_q_values[legal_actions]
                # action = legal_actions[np.argmax(legal_q_values)]
                action = self.joint_index_to_joint_actions[np.argmax(joint_q_values)][player]
                # print("Joint state has been seen: ", ''.join(map(str, time_step.observations["global_state"][0][:])) in self._seen_observations, "  Action outputted: ", action)
                probs[action] = 1.0
                """
            else:
                probs = np.zeros(self._num_actions)
                legal_actions = time_step.observations["legal_actions"][player]
                """
                if not is_evaluation and np.random.rand() < self.get_epsilon_fine_tune: 
                    # action = np.random.choice(legal_actions)
                    probs[legal_actions] = 1.0 / len(legal_actions)
                    action = utils.random_choice(list(range(self._num_actions)), probs)
                else:
                    q_values = self.session.run(self._q_values, feed_dict={self._info_state_ph: info_state})[0]
                    # print("Q values for player {}".format(player), q_values)


                    legal_q_values = q_values[legal_actions]

                    # action_probs = np.exp(legal_q_values) / np.sum(np.exp(legal_q_values))
                    # action = utils.random_choice(legal_actions, action_probs)

                    action = legal_actions[np.argmax(legal_q_values)]
                    probs[action] = 1.0
                """ 
                q_values = self.session.run(self._q_values, feed_dict={self._info_state_ph: info_state})[0]
                legal_q_values = q_values[legal_actions]
                action = legal_actions[np.argmax(legal_q_values)]

                probs[action] = 1.0
            # TODO: Remove this. THis is for experiment 
            # if is_evaluation:
            #     self.states_seen_in_evaluation.append(''.join(map(str, time_step.observations["info_state"][self.player_id])))
        else:
            action = None
            probs = []

        return rl_agent.StepOutput(action=action, probs=probs)

    def _initialize(self):
        # initialization_weights = tf.group(
        #     *[var.initializer for var in self._policy_variables])
        # initialization_opt = tf.group(
        #     *[var.initializer for var in self._policy_optimizer.variables()])
        initialization_q_weights = tf.group(
            *[var.initializer for var in self._q_variables])
        initialization_q_opt = tf.group(
            *[var.initializer for var in self._q_optimizer.variables()])
        initialization_q_target_weights = tf.group(
            *[var.initializer for var in self._target_q_variables])
        initialization_policy_weights =tf.group(
            *[var.initializer for var in self._output_policy.variables[:]]
        )
        initialization_policy_opt = tf.group(
            *[var.initializer for var in self._policy_extraction_optimizer.variables()]
        )
    
        # initialization_weights,
        # initialization_opt,
        self.session.run(
            tf.group(*[
                initialization_q_weights, 
                initialization_q_opt,
                initialization_q_target_weights,
                initialization_policy_weights,
                initialization_policy_opt
            ]))
    
    def logits_to_action(self, logits):
        assert not np.isnan(logits).any()
        action_probs = np.exp(logits) / np.sum(np.exp(logits))
        # action_probs = action_probs.eval()  # to numpy
        if np.isnan(action_probs).any():
            action_probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
            assert not np.isnan(action_probs).any()
        action = np.random.choice(self._num_actions, p=action_probs)
        return action, action_probs

    def cumsum(self, x, discount):
        vals = [None for _ in range(len(x))]
        curr = 0 
        for i in range(len(x) - 1, -1, -1):
            vals[i] = curr = x[i] + discount  * curr
        return vals

    def add_trajectory(self, trajectory, action_trajectory, override_symmetric=False):
        """Trajectory is a list of timesteps, Action_trajectory is a list of lists representing joint actions. If it is a single player playing an action, 
            it will be a list of length 1 lists. """
        """ Adds the trajectory consisting only of transitions relevant to the current player (only self.player_id if turn-based or all of them if simultaneous)"""
        if self._fine_tune_mode: 
            # This add_trajectory is used by the joint_wrapper
            assert False
            # self._fine_tune_module.add_trajectory(trajectory, action_trajectory, override_symmetric)
            return 

        rewards_to_go = [np.zeros(self.num_players) for _ in range(len(trajectory) - 1)]

        for p in range(self.num_players):
            rewards = np.reshape(np.array([timestep.rewards[p] for timestep in trajectory[1:]]), (-1, 1))
            curr_rewards = self.cumsum(rewards, self.discount_factor)
            assert len(rewards_to_go) == len(curr_rewards)
            for i, val in enumerate(rewards_to_go):
                val[p] = curr_rewards[i]

        for i in range(len(trajectory) - 1):
            if not self._is_turn_based:
                # NOTE: If is_symmetric, then add_transition will add observations/actions from BOTH players already
                # NOTE: Also insert action_trajectory[i+1]. If it is the last i, then we let action be 0 because it won't be used anyway
                next_action = action_trajectory[i+1] if (i+1) < len(action_trajectory) else [0 for _ in range(self.num_players)] 
                self.add_transition(trajectory[i], action_trajectory[i], trajectory[i+1], next_action, ret=rewards_to_go[i], override_symmetric=override_symmetric) 
            else:
                # Individual player's move
                # NOTE: Assume that anything called using add_trajectory already filters out for the relevant transitions  
                player = trajectory[i].observations["current_player"]
                next_action = action_trajectory[i+1] if (i+1) < len(action_trajectory) else [0 for _ in range(self.num_players)] 
                self.add_transition(trajectory[i], action_trajectory[i], trajectory[i+1], next_action, ret=rewards_to_go[i], gae=gae[i], override_player=[player])
                """player = trajectory[i].observations["current_player"]

                action = [0 for _ in range(self.num_players)]
                next_action = [0 for _ in range(self.num_players)]
                
                action[player] = action_trajectory[i][0]  # Only one player acted here
                
                # Simply indexing i+1 for next state is incorrect. No guarantees it is this player's move or the final. So, loop until find the next state with player 
                next_player_timestep_index = None
                for j in range(i+1, len(trajectory)):
                    if trajectory[j].observations["current_player"] == player or trajectory[j].last():
                        next_player_timestep_index = j 
                        break 
                if next_player_timestep_index:
                    next_action[player] = action_trajectory[next_player_timestep_index][0] if next_player_timestep_index < len(action_trajectory) else 0
                    curr_policy.add_transition(trajectory[i], action, trajectory[next_player_timestep_index], next_action, ret=rewards_to_go[i], override_player=[player]) """
            
    def add_transition(self, prev_time_step, prev_action, time_step, action, ret, override_symmetric=False, override_player=[]):
        """Adds the new transition using `time_step` to the replay buffer.

        Adds the transition from `self._prev_timestep` to `time_step` by
        `self._prev_action`.

        Args:
          prev_time_step: prev ts, an instance of rl_environment.TimeStep.
          prev_action: list of ints, joint action taken at `prev_time_step`.
          time_step: current ts, an instance of rl_environment.TimeStep.
          action: list of ints, joint action taken at timestep 
          ret: return of the trajectory associated with 
          override_player: player this trajectory is associated with. If empty, then default.
        """
        player_list = [i for i in range(self.num_players)] if self.symmetric and not override_symmetric else [self.player_id]
        if len(override_player) > 0:
            player_list = override_player

        if self.joint_action: 
            o = prev_time_step.observations["global_state"][0][:] if len(prev_time_step.observations["global_state"]) > 0 else np.concatenate(prev_time_step.observations["info_state"])
            r = sum(time_step.rewards)  # NOT accounting for rewards_joint here
            rewards_to_go = sum(ret)

            store_action = int(self._calculate_joint_action_index(np.array(prev_action).reshape(1, -1))[0][0])
            store_next_action =  int(self._calculate_joint_action_index(np.array(action).reshape(1, -1))[0][0])
            next_o = time_step.observations["global_state"][0][:] if len(time_step.observations["global_state"]) > 0 else np.concatenate(time_step.observations["info_state"])
            # each element in legal_actions mask is a mask for each player. And each mask is self._num_actions long with 0 or 1. 
            
            all_legal_actions = time_step.observations["legal_actions"]
            # Create a length num_actions ** num_players array of 0's 
            legal_actions_mask = [0 for i in range(self._num_actions ** self.num_players)]
            for i in range(len(legal_actions_mask)):
                joint_action = self.joint_index_to_joint_actions[i]
                legal_actions_mask[i] = 1 if all([joint_action[i] in all_legal_actions[i] for i in range(self.num_players)]) else 0
            
            transition = Transition(
                    info_state=o,
                    action=store_action,
                    reward=r, 
                    next_info_state=next_o,
                    next_action = store_next_action,
                    is_final_step=float(time_step.last()),
                    legal_actions_mask=legal_actions_mask,
                    rewards_to_go=rewards_to_go
                )
            self._seen_observations.add(''.join(map(str, o)))
            self._replay_buffer.add(transition)
            # for each element, get the marginal player indices 
        else:
            for p in player_list:
                o = prev_time_step.observations["info_state"][p][:]

                assert not self.rewards_joint  # because gae is not valid

                r = sum(time_step.rewards) if self.rewards_joint else time_step.rewards[p] # WELFARE
                rewards_to_go = sum(ret) if self.rewards_joint else ret[p]

                # NOTE: We want to keep all the transitions consistent...as if it was from one player's perspective.
                # So, we need to switch the actions for consistency (this implementation only works for 2-player games)
                # If we didn't switch, then the same action for the same observation can yield different results depending on which player we took that transition from 

                # Since the step() function assumes by symmetry that observations come from player0, we need to make sure that all 
                # transitions are from player0's perspective, meaning the action applied to the observed player's observation must come first

                store_action = prev_action[p]
                store_next_action = action[p]

                legal_actions = (time_step.observations["legal_actions"][p])
                legal_actions_mask = np.zeros(self._num_actions)
                legal_actions_mask[legal_actions] = 1.0

                transition = Transition(
                    info_state=(
                        prev_time_step.observations["info_state"][p][:]),
                    action=store_action,
                    reward=r, 
                    next_info_state=time_step.observations["info_state"][p][:],
                    next_action = store_next_action,
                    is_final_step=float(time_step.last()),
                    legal_actions_mask=legal_actions_mask,
                    rewards_to_go=rewards_to_go, 
                )

                self._replay_buffer.add(transition)


    def learn(self):
        """Compute the loss on sampled transitions and perform a Q-network update.

        If there are not enough elements in the buffer, no loss is computed and
        `None` is returned instead.

        Returns:
          The average loss obtaine d on this batch of transitions or `None`.
        """
        assert not self._fine_tune_mode
        loss = 0
        length = len(self._replay_buffer)
        dataset = self._replay_buffer.sample(length)  # samples without replacement so take the entire thing. Random order
        indices = list(range(length))

        rtg_all = np.array([transition.rewards_to_go for transition in dataset])
        rtg_mean = np.mean(rtg_all)
        rtg_std = np.std(rtg_all)
        # normalize_rtg = (rtg_all - rtg_mean) / rtg_std
        # rtg_shift = min(normalize_rtg)  # subtract this value to get our target

        obs_to_actions = {}
        for t in dataset:
            curr_obs = t.info_state
            key = ''.join(map(str, curr_obs))
            action_counts = obs_to_actions.get(key, [0 for _ in range(self._num_actions if not self.joint_action else self._num_actions ** self.num_players)])
            action_counts[t.action] += 1
            obs_to_actions[key] = action_counts
 
        # Train Q-values
        value_steps_total = 0 
        epoch = 0
        while value_steps_total < (self.training_steps): # if not self._fine_tuning_mode else self.training_steps / 5):
            epoch += 1
            i, batch, bellman_loss_total, loss_total = 0, 0, 0, 0
            dataset = random.sample(dataset, len(dataset))
            while i < length:
                transitions = dataset[i: min(length, i+self.batch)] 
                
                # with tf.device(self.device):
                info_states = [t.info_state for t in transitions]
                actions = [[t.action] for t in transitions]
                rewards = [t.reward for t in transitions]
                next_info_states = [t.next_info_state for t in transitions]
                next_actions = [[t.next_action] for t in transitions]
                done = [t.is_final_step for t in transitions]
                legal_actions_mask = [t.legal_actions_mask for t in transitions]\
                # We normalize these rewards go to for training offline (as we have acccess to all rtg). Fine-tuning uses a different buffer and is NOT normalized
                rewards_to_go = [((t.rewards_to_go - rtg_mean) / rtg_std) for t in transitions] # [t.rewards_to_go for t in transitions] #  - rtg_shift for t in transitions] [t.rewards_to_go for t in transitions] #

                start = time.time()
                loss, bellman_loss, _, q_values = self.session.run(
                    [self._q_loss, self._bellman_loss, self._q_learn_step, self._q_values],
                    feed_dict={
                        self._info_state_ph: info_states,
                        self._action_ph: actions, 
                        self._reward_ph: rewards,
                        self._next_info_state_ph: next_info_states, 
                        self._next_action_ph: next_actions,
                        self._is_final_step_ph: done, 
                        self._legal_actions_mask_ph: legal_actions_mask,
                        self._rewards_to_go_ph: rewards_to_go,
                        self._fine_tune_mode_ph: False
                    })
                # print("R-BVE learn time: ", time.time() - start)
                # print("Example check: ", obs_to_actions[''.join(map(str, info_states[0]))], q_values[0])

                value_steps_total += 1
                i += self.batch
                batch += 1
                loss_total += loss
                bellman_loss_total += bellman_loss

                if value_steps_total % self.update_target_network_every == 0 and value_steps_total != 0:
                    self.session.run(self._update_target_network)

            if epoch % 20 == 0 and batch > 0:
                print("Epoch {} had average loss of {} and bellman loss of {} for value training of {} total steps".format(epoch, loss_total / float(batch), bellman_loss_total / float(batch), value_steps_total))
        
        return loss 

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
