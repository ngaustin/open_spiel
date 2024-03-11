"""
Main wrapper responsible for training a ratio model (encodes the policy-induced state-occupancy distribution). This will be used in 
response training and coupled with the value network training loss, using the COP-TD value loss for off-policy, offline learning.
"""

import collections
import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import time

from open_spiel.python import simple_nets


class TargetExtractor:
    def __init__(self, session, data, behavior_policies, target_policies, target_extractor_parameters):
        """
        TargetExtractor is a module that predicts the state occupancy ratio given a state for a particular set of 
        behavior and target policy pairing. We note that target_policies gives each player one policy, meaning that 
        we are not accounting for mixed profiles. Instead, we will be training best responses to pure strategies 
        and using policy mixing with the ratio model as our opponent policy classifier.

        behavior_policies: a list of policy objects denoting the policies that characterize the dataset
        target_policies: a list of policy objects denoting the policies that are being trained
        target_extractor_parameters: parameters associated with the target extractor
        """
        self._session = session
        self._data = data
        self._num_players = target_extractor_parameters["num_players"]
        self._num_actions = target_extractor_parameters["num_actions"]
        self._relevant_players = target_extractor_parameters["relevant_players"]  # this is a list indicating which player's transitions to train off of

        self._ratio_model_discount = target_extractor_parameters["ratio_model_discount"]
        self._ratio_model_soft_normalization = target_extractor_parameters["ratio_model_normalization_strength"]
        self._ratio_model_batch_size = target_extractor_parameters["ratio_model_batch_size"]
        self._ratio_model_lr = target_extractor_parameters["ratio_model_lr"]
        self._num_learn_steps = target_extractor_parameters["ratio_model_num_learn_steps"]

        self._model_width = target_extractor_parameters["ratio_model_width"]
        self._model_num_layers = target_extractor_parameters["ratio_model_num_layers"]
        self._model_state_size = target_extractor_parameters["ratio_model_state_size"]
        self._update_target_every = target_extractor_parameters["update_target_every"]

        self._model_shape = [self._model_width] * self._model_num_layers

        self._ratio_model = simple_nets.MLP(input_size=self._model_state_size, hidden_sizes=self._model_shape, output_size=1)
        self._ratio_target_model = simple_nets.MLP(input_size=self._model_state_size, hidden_sizes=self._model_shape, output_size=1)
        self._behavior_policies = behavior_policies
        self._target_policies = target_policies 
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._ratio_model_lr)
        self._steps = 0

        self.initialize()

        self._update_target_network = self._create_target_network_update_op(self._ratio_model, self._ratio_target_model)

        self._state_ph = tf.placeholder(shape=[None, self._model_state_size], dtype=tf.float32, name="state_ph")
        self._next_state_ph = tf.placeholder(shape=[None, self._model_state_size], dtype=tf.float32, name="next_state_ph")
        self._observation_ph = tf.placeholder(shape=[None, self._num_players, self._observation_size], dtype=tf.float32, name="observation_ph")
        self._action_ph = tf.placeholder(shape=[None, self._num_players, 1], dtype=tf.float32, name="action_ph")
        self._legal_actions_mask_ph = tf.placeholder(shape=[None, self._num_players, self._num_actions], dtype=tf.float32, name="legal_actions_mask_ph")
        self._relevant_players_mask_ph = tf.placeholder(shape=[None, self._num_players], dtype=tf.float32, name="relevant_players_mask_ph")

        # Computing the COP-TD discounted learning loss 
        state_ratios = tf.stop_gradient(self._ratio_target_model(self._state_ph))  # shape is [N, 1]
        next_state_ratios = self._ratio_model(self._next_state_ph)  # shape is [N, 1]

        numerator = np.ones((tf.shape(self._state_ph)[0], 1))  # shape is [N, 1]
        denominator = np.ones((tf.shape(self._state_ph)[0], 1))

        for i in range(self._num_players):
            # Assemble all of the observations relevant to player i 
            player_observations = self._observation_ph[:, i, :] # shape is [N, observation_size]
            player_actions = self._action_ph[:, i, :]
            player_legal_actions_mask = self._legal_actions_mask_ph[:, i, :]

            # Query for their probability of action 
            target_action_probabilities = self._target_policies[i].probabilities_with_actions(player_observations, player_actions, player_legal_actions_mask, numpy=True)

            # Multiply into numerator multiply when relevant_players_mask_ph is 1 for i-th player. Otherwise, multiply by 1.
            numerator = numerator * (target_action_probabilities * self._relevant_players_mask_ph[:, i] + (1 - self._relevant_players_mask_ph[:, i])) 

            # Do the same for behavior policy 
            behavior_action_probabilities = self._behavior_policies[i].probabilities_with_actions(player_observations, player_actions, player_legal_actions_mask, numpy=True)
            denominator = denominator * behavior_action_probabilities
    

        # Compute the mean squared error of the loss function and save 
        action_probability_ratio = numerator / denominator  # shape is [N, 1]
        target_value = self._ratio_model_discount * state_ratios * action_probability_ratio + (1 - self._ratio_model_discount)
        ratio_loss = tf.reduce_mean(tf.square(target_value - next_state_ratios))


        # Computing the Soft Ratio Normalization 
        state_ratios_soft = self._ratio_model(self._state_ph)  # shape is [N, 1]
        
        # Create an [N, N] matrix where the diagonal elements are 0 while off-diagonals are 1.
        accumulator_matrix = tf.ones([tf.shape(state_ratios_soft)[0]] * 2) - tf.linalg.tensor_diag([1] * tf.shape(state_ratios_soft)[0])

        # Calculate the soft ratio normalization loss
        summed_ratios = tf.linalg.matmul(accumulator_matrix, tf.stop_gradient(state_ratios_soft)) # shape is [N, 1]
        weighted_by_sum_ratios = (summed_ratios - 1) * state_ratios_soft # shape is [N, 1]
        soft_ratio_normalization_loss = tf.reduce_mean(weighted_by_sum_ratios * state_ratios_soft)

        self._loss = ratio_loss + self._ratio_model_soft_normalization * soft_ratio_normalization_loss
        self._learn_step = self._optimizer.minimize(self._loss)

    def _initialize(self):
        initialization_weights = tf.group(*[var.initializer for var in self._ratio_model.variables[:]])
        initialization_target_weights = tf.group(*[var.initializer for var in self._ratio_target_model.variables[:]])
        initialization_opt = tf.group(*[var.initializer for var in self._optimizer.variables()])

        self._session.run(tf.group(*[initialization_weights, initialization_target_weights, initialization_opt]))

    def _create_target_network_update_op(self, source_network, target_network):
        """Create TF ops copying the params of the Q-network to the target network.

        Args:
        q_network: A q-network object that implements provides the `variables`
                    property representing the TF variable list.
        target_q_network: A target q-net object that provides the `variables`
                            property representing the TF variable list.

        Returns:
        A `tf.Operation` that updates the variables of the target.
        """
        variables = source_network.variables[:]
        target_variables = target_network.variables[:]

        return tf.group([
            tf.assign(target_v, v)
            for (target_v, v) in zip(target_variables, variables)
        ])

    def predict_ratio(self, states):
        """
        Given the behavior policies (of the dataset) and target policies (the policies currently being trained), compute the 
        state-occupancy ratios from the current ratio model.

        states: a numpy or tf array of size [N, self._model_state_size]

        Return: a tf array of size [N, 1] representing the occupancy ratios of each state
        """
        if not tf.is_tensor(states):
            states = tf.convert_to_tensor(states)

        return tf.stop_gradient(self._ratio_model(states))  # Size [N, 1]

    def train_ratio_model(self):
        logging.info("Ratio model training info: ")
        return 
    
    def execute_learn_step_ratio_model(self, batch):
        """
        Given a batch of Transition objects, execute one learning step. 

        batch: a list of Transition objects

        Return: loss value of the learn step
        """

        info_states = [t.info_state for t in batch]
        global_states = [t.global_states for t in batch]
        actions = [t.action for t in batch]
        next_global_states = [t.next_global_states for t in batch]
        legal_actions_mask = [t.legal_actions_mask for t in batch]
        # relevant_players = [t.relevant_players for t in batch]

        info_states = 
        global_states = 
        actions = 
        next_global_states = 
        legal_actions_mask = 
        # relevant_players = 

        for transition in batch:
            # Make sure there is non-zero overlap between the transition's relevant players and self._relevant_players
            relevant_players_in_transition = [p for p in self._relevant_players if p in transition.relevant_players]
            assert len(relevant_players_in_transition) >= 1
            
            for p in relevant_players_in_transition:
                # For each self._relevant_players, add the info_states, global_states etc. according to the placeholder shape 
                info_states.append(transition.info_state)  # should be joint
                actions.append(transition.action)  # should be joint
                legal_actions_mask.append(transition.legal_actions_mask)  # should be joint 
                global_states.append(transition.global_states[p])
                next_global_states.append(transition.next_global_states[p])

                # If done == 1, we need special handling. 
                # Given the player p in question, we need to figure out the "relevant players" that contributed to this transition. 
                # More concretely, another player would only contribute to the probability of this transition iff it acted after player p and before episode termination.
                
                relevant_players_mask


            

            # If the done == 1, we need special handling. For each self._relevant_players, filter. 

        # TODO: Filter out the global and next_global_states based on relevant_players
        # TODO: If DONE field is 1, we need some special handling! 

        relevant_players_mask = np.zeros([len(info_states), self._num_players, self._num_actions])

        for i in range(len(info_states)):
            for p in relevant_players[i]:
                # relevant_observations[i][p] = info_states[i][p]
                # relevant_actions[i][p] = actions[i][p]
                # relevant_legal_actions_mask = legal_actions_mask[i][p]
                relevant_players_mask[i][p] = 1

        
        # Pass into the self._session and request self._loss and self._learn_step
        self._session.run([self._loss, self._learn_step],
            feed_dict={
                self._state_ph: global_states,
                self._next_state_ph: next_global_states,
                self._observation_ph: info_states, 
                self._action_ph: actions,
                self._legal_actions_mask_ph: legal_actions_mask,
                self._relevant_players_mask_ph: relevant_players_mask
            })

        self._steps += 1

        # Update target network 
        if self._steps % self._update_target_every == 0:
            self._session.run(self._update_target_network)
            self._steps = 0


    