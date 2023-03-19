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

from open_spiel.python import rl_agent
from open_spiel.python import simple_nets
from open_spiel.python.utils.replay_buffer import ReplayBuffer

# Temporarily disable TF2 behavior until code is updated.
tf.disable_v2_behavior()

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state is_final_step")

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
                 num_players):
        """Initialize the DQN agent."""

        # This call to locals() is used to store every argument used to initialize
        # the class instance, so it can be copied with no hyperparameter change.
        self._kwargs = locals()
        self.session = consensus_kwargs["session"]

        self.player_id = player_id
        self._num_actions = num_actions

        self.observations = []  # list of observations
        self.q_values = []  # for each obs, we have a dictionary that maps each action to a DICTIONARY of q values.
                            # Maps a frozenset of other player actions to q value
        self.trajectory = []  # list of tuples. Each tuple is (prev observation, prev action, reward, observation, done)

        self.epochs = consensus_kwargs["training_epochs"]
        self.minimum_entropy = consensus_kwargs["minimum_entropy"]
        self.lr = consensus_kwargs["deep_network_lr"]
        self.policy_lr = consensus_kwargs["deep_policy_network_lr"]
        self.batch = consensus_kwargs["batch_size"]
        self.hidden_layer_size = consensus_kwargs["hidden_layer_size"]
        self.n_hidden_layers = consensus_kwargs["n_hidden_layers"]
        self.joint = consensus_kwargs["joint"]
        self.rewards_joint = consensus_kwargs["rewards_joint"]
        self.alpha = consensus_kwargs["alpha"]
        self.boltzmann = consensus_kwargs["boltzmann"]
        self.update_target_network_every = consensus_kwargs["update_target_every"]
        self.discount_factor = .99
        # self.tau = 1e-2

        # Look at above note about marginalizing Q-values for the joint action
        self.player_marginal_indices = [[] for _ in range(num_actions)]  # action index -> list of indices for marginalization
        self.joint_index_to_marginal_index = {}
        max_num_actions = self._num_actions ** num_players

        all_actions = list(product(list(range(num_actions)), repeat=num_players))
        print("THERE ARE A TOTAL OF {} ACTIONS IN JOINT SPACE".format(len(all_actions)))
        print("Actions here: {}".format(all_actions))
        for joint_action in all_actions:
            joint_action_index = self._calculate_joint_action_index(np.array(joint_action).reshape(1, -1))
            marginalized_action = joint_action[self.player_id]
            self.player_marginal_indices[marginalized_action].append(int(joint_action_index[0][0]))
            self.joint_index_to_marginal_index[joint_action_index[0][0]] = marginalized_action
        
        print("Mappings for actions to joint indices: {}".format(self.player_marginal_indices))
        print("Mappings joint index to actions: {}".format(self.joint_index_to_marginal_index))
        """
        for a in range(num_actions):
            start = a * (num_actions ** self.player_id)
            curr_index_list = []
            while start < max_num_actions:
                finish = start + (num_actions ** self.player_id)
                curr_index_list.extend(list(range(start, finish)))
                start += (num_actions) ** (self.player_id + 1)
            self.player_marginal_indices.append(curr_index_list)
        """
        self.player_marginal_indices = np.array(self.player_marginal_indices, dtype=np.dtype(int))
        
        self.layer_sizes = [self.hidden_layer_size] * self.n_hidden_layers

        # Initialize replay
        self.replay_buffer = ReplayBuffer(np.inf)

        # Initialize the FF network 
        # self.policy_net = simple_nets.MLP(state_representation_size,
                                   # self.layer_sizes, self._num_actions)

        self._q_network = simple_nets.MLP(state_representation_size, 
                                    self.layer_sizes, self._num_actions ** (num_players))

        self._target_q_network = simple_nets.MLP(state_representation_size,
                                             self.layer_sizes, num_actions ** (num_players))

        # self._policy_variables = self.policy_net.variables[:] 
        self._q_variables = self._q_network.variables[:]
        self._target_q_variables = self._target_q_network.variables[:]
        
        # Create placeholders
        self._info_state_ph = tf.placeholder(
            shape=[None, state_representation_size],
            dtype=tf.float32,
            name="info_state_ph")
        self._action_ph = tf.placeholder(
            shape=[None, num_players], dtype=tf.int32, name="action_ph")
        self._reward_ph = tf.placeholder(
            shape=[None], dtype=tf.float32, name="reward_ph")
        self._is_final_step_ph = tf.placeholder(
            shape=[None], dtype=tf.float32, name="is_final_step_ph")
        self._next_info_state_ph = tf.placeholder(
            shape=[None, state_representation_size],
            dtype=tf.float32,
            name="next_info_state_ph")
        self._joint_action_index_ph = tf.placeholder(
            shape=[None, 1], dtype=tf.int32, name="joint_action_ph")

        # Define placeholder functions and operations

        # Q values for current states
        self._q_values = self._q_network(self._info_state_ph)

        # Defining Q value targets for MSE loss
        self._target_q_values = self._target_q_network(self._next_info_state_ph)

        next_q_double = self._q_network(self._next_info_state_ph)
        max_next_a = tf.math.argmax(tf.stop_gradient(next_q_double), axis=-1)
        max_next_q = tf.gather(self._target_q_values, max_next_a, axis=1, batch_dims=1)

        # max_next_q = tf.reduce_max(tf.stop_gradient(self._target_q_values), axis=-1)

        self._update_target_network = self._create_target_network_soft_update_op(
                                        self._q_network, self._target_q_network)

        target = (self._reward_ph + (1 - self._is_final_step_ph) * self.discount_factor * max_next_q)

        minimize_q_function = tf.math.reduce_logsumexp(self._q_values, axis=1) # tf.gather(self._q_values, self._joint_action_index_ph, axis=1, batch_dims=1)

        maximize_q_function = tf.gather(self._q_values, self._joint_action_index_ph, axis=1, batch_dims=1)

        # True predictions for the given joint actions
        predictions = tf.gather(self._q_values, self._joint_action_index_ph, axis=1, batch_dims=1)# action_indices)
        target = tf.reshape(target, [-1, 1])

        # Calculate CQL Q-network loss
        loss_class = tf.losses.mean_squared_error
        self._bellman_loss = tf.reduce_mean(loss_class(labels=target, predictions=predictions))
        self._q_loss = .5 * self._bellman_loss + self.alpha * tf.reduce_mean(minimize_q_function - maximize_q_function)

        # Retrieve logits for current policy given states
        """
        self._logits = self.policy_net(self._info_state_ph)

        # Marginalize the Q-values given the player ID and their marginal actions
        # TODO: Consider normalization here of some kind
        action_indices = tf.reshape(self._action_ph[:, self.player_id], [-1])
        marginalized_indices_tensor = tf.gather(self.player_marginal_indices, action_indices)
        marginalized_q_values = tf.gather(tf.stop_gradient(self._q_values), marginalized_indices_tensor, axis=1, batch_dims=1)

        # Find the max of the marginalized Q-values
        self.max_marginalized_q_values = tf.reduce_max(marginalized_q_values, axis=1)

        self.max_of_q_values = tf.reduce_max(self.max_marginalized_q_values)
        self.mean_of_q_values = tf.reduce_mean(self.max_marginalized_q_values)

        # Gather logits corresponding to actions taken
        action_indices = tf.stack(
             [tf.range(tf.shape(self._q_values)[0]), self._action_ph[:, self.player_id]], axis=-1)
        # logit_specific = tf.gather_nd(self._logits, action_indices)

        # Calculate entropy for each of the states
        self.probs = tf.nn.softmax(self._logits, axis=1)
        probs_specific = tf.gather_nd(self.probs, action_indices)
        entropy = tf.math.reduce_sum(self.probs * tf.math.log(self.probs), axis=1)
        self.average_entropy = tf.reduce_mean(entropy)
        # entropy = tf.reshape(entropy, [-1, 1])

        # Calculate policy loss
        print("Shapes of stuff:", probs_specific.shape, self.max_marginalized_q_values.shape, entropy.shape)
        self._policy_loss = -tf.reduce_mean(self.max_marginalized_q_values * tf.math.log(probs_specific))#  - entropy)

        # Initialize Adam Optimizer 
        self._policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.policy_lr)
        self._policy_learn_step = self._policy_optimizer.minimize(self._policy_loss)
        """

        self._q_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self._q_learn_step = self._q_optimizer.minimize(self._q_loss)


        self.running_steps = 0
        self.running_not_seen_steps = 0

        self._initialize()

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
            tf.assign(target_v, v)# self.tau * v + (1 - self.tau) * target_v)
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
        # TODO: Get rid of this because it's for testing
        self.running_steps += 1
        if (not time_step.last()) and (
                time_step.is_simultaneous_move() or
                self.player_id == time_step.current_player()):

            """
            if self.joint:
                info_state = np.hstack(time_step.observations["info_state"])
            else:
                info_state = time_step.observations["info_state"][self.player_id]
            """
            info_state = time_step.observations["info_state"][self.player_id]
            
            info_state = np.reshape(info_state, [1, -1])

            # TODO: Use the q network, take the max, then backwards marginalize! 
            joint_q_values = self.session.run(self._q_values, feed_dict={self._info_state_ph: info_state})[0]
            
            # print("PLAYER MARGINAL INDICES: ", self.player_marginal_indices)
            # print("Q vals: ", joint_q_values)
            actions_mapped_to_values = np.take(joint_q_values, self.player_marginal_indices)
            # print("Act to values: ", actions_mapped_to_values)
            marginalized_q_values = np.max(actions_mapped_to_values, axis=1)
            # print("Q vals: ", marginalized_q_values)

            # exps = np.exp(self.boltzmann * marginalized_q_values) + 1e-6
            # probs = exps / np.sum(exps)

            # if np.isnan(probs).any() or np.sum(probs) != 1:
            probs = np.zeros(self._num_actions)
            action = np.argmax(marginalized_q_values)
            probs[action] = 1.0
            # else:
            #     action = np.random.choice(self._num_actions, 1, p=probs)[0]

            # logits = self.session.run(self._logits, feed_dict={self._info_state_ph: info_state})[0]
            # print("Logits: ", logits)
            
            # action, probs = self.logits_to_action(logits)
            return rl_agent.StepOutput(action, probs)
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
        
    
        # initialization_weights,
        # initialization_opt,
        self.session.run(
            tf.group(*[
                initialization_q_weights, 
                initialization_q_opt,
                initialization_q_target_weights
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


    def add_transition(self, prev_time_step, prev_action, time_step, ret):
        """Adds the new transition using `time_step` to the replay buffer.

        Adds the transition from `self._prev_timestep` to `time_step` by
        `self._prev_action`.

        Args:
          prev_time_step: prev ts, an instance of rl_environment.TimeStep.
          prev_action: int, action taken at `prev_time_step`.
          time_step: current ts, an instance of rl_environment.TimeStep.
        """
        o = prev_time_step.observations["info_state"][self.player_id][:]
        # TODO: Consider not using welfare but rather individual rewards 
        r = sum(time_step.rewards) if self.rewards_joint else time_step.rewards[self.player_id] # WELFARE

        transition = Transition(
            info_state=(
                prev_time_step.observations["info_state"][self.player_id][:]),
            action=prev_action,
            reward=r, 
            next_info_state=time_step.observations["info_state"][self.player_id][:],
            is_final_step=float(time_step.last())
        )

        self.replay_buffer.add(transition)


    def learn(self):
        """Compute the loss on sampled transitions and perform a Q-network update.

        If there are not enough elements in the buffer, no loss is computed and
        `None` is returned instead.

        Returns:
          The average loss obtaine d on this batch of transitions or `None`.
        """
        length = len(self.replay_buffer)
        dataset = self.replay_buffer.sample(length)  # samples without replacement so take the entire thing. Random order
        indices = list(range(length))

        # Train Q-values in joint action space
        value_steps_total = 0 
        for j in range(self.epochs):
            i, batch, bellman_loss_total, loss_total = 0, 0, 0, 0
            dataset = random.sample(dataset, len(dataset))
            while i < length:
                transitions = dataset[i: min(length, i+self.batch)] 
                info_states = [t.info_state for t in transitions]
                actions = [t.action for t in transitions]
                rewards = [t.reward for t in transitions]
                next_info_states = [t.next_info_state for t in transitions]
                done = [t.is_final_step for t in transitions]

                joint_action_index = self._calculate_joint_action_index(np.array(actions))

                loss, bellman_loss, _ = self.session.run(
                    [self._q_loss, self._bellman_loss, self._q_learn_step],
                    feed_dict={
                        self._info_state_ph: info_states,
                        self._action_ph: actions, 
                        self._reward_ph: rewards,
                        self._next_info_state_ph: next_info_states, 
                        self._is_final_step_ph: done,
                        self._joint_action_index_ph: joint_action_index
                    })

                value_steps_total += 1
                i += self.batch
                batch += 1
                loss_total += loss
                bellman_loss_total += bellman_loss

                if value_steps_total % self.update_target_network_every == 0:
                    self.session.run(self._update_target_network) 

            print("Epoch {} had average loss of {} and bellman loss of {} for value training of {} total steps".format(j, loss_total / float(batch), bellman_loss_total / float(batch), value_steps_total))
        
        """
        # Convert to marginalized policy
        for j in range(1000):
            i, batch, loss_total, entropy_total = 0, 0, 0, 0
            dataset = random.sample(dataset, len(dataset))
            while i < length:
                transitions = dataset[i: min(length, i+self.batch)] 
                info_states = [t.info_state for t in transitions]
                actions = [t.action for t in transitions]
                rewards = [t.reward for t in transitions]
                next_info_states = [t.next_info_state for t in transitions]
                done = [t.is_final_step for t in transitions]

                q, logits, probs, entropy, max_q_vals, mean_q_vals, loss, _ = self.session.run(
                    [self._q_values, self._logits, self.probs, self.average_entropy, self.max_of_q_values, self.mean_of_q_values, self._policy_loss, self._policy_learn_step],
                    feed_dict={
                        self._info_state_ph: info_states,
                        self._action_ph: actions, 
                        self._reward_ph: rewards
                    }
                )
                # print("Probs: {}, Entropy: {}".format(probs, entropy))
                # if i == 0:
                #     print("Max of the Q_vals: {}, Mean of the Q Vals: {}".format(max_q_vals, mean_q_vals))
                i += self.batch
                batch += 1
                loss_total += loss 
                entropy_total += entropy
            
            print("Epoch {} had average loss of {} and entropy of {} for policy training".format(j, loss_total / float(batch), entropy_total / float(batch)))
            if -entropy_total / float(batch) < self.minimum_entropy:
                break
        """
        return loss 

    def get_weights(self):
        # TODO: Implement this
        return [0]

    def get_max_q_value(self):
        max_value = -np.inf
        # for action_to_q_dict in self.q_values:
        #     for other_action_dict in action_to_q_dict.values():
        #         max_value = max(max_value, max(other_action_dict.values()))
        action_to_q_dict = self.q_values[0]  # the first added state is the initial state
        for other_action_dict in action_to_q_dict.values():
          max_value = max(max_value, max(other_action_dict.values()))
        return max_value

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
