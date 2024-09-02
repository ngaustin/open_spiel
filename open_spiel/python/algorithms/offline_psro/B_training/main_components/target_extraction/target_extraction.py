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

import matplotlib.pyplot as plt


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
        # data is a LIST of datasets of same length as self._relevant_players. Note that data MUST include the initial step dataset as well (otherwise probabilities will not be propagated appropriately)
        self._data = data  
        self._num_players = target_extractor_parameters["num_players"]
        self._num_actions = target_extractor_parameters["num_actions"]
        self._relevant_players = target_extractor_parameters["relevant_players"]  # this is a list indicating which player's transitions to train off of

        self._ratio_model_discount = target_extractor_parameters["ratio_model_discount"]
        self._ratio_model_soft_normalization = target_extractor_parameters["ratio_model_normalization_strength"]
        self._ratio_model_batch_size = target_extractor_parameters["ratio_model_batch_size"]
        self._ratio_model_lr = target_extractor_parameters["ratio_model_lr"]

        self._model_width = target_extractor_parameters["ratio_model_width"]
        self._model_num_layers = target_extractor_parameters["ratio_model_num_layers"]
        self._model_state_size = target_extractor_parameters["ratio_model_state_size"]
        self._observation_size = target_extractor_parameters["info_state_size"]
        self._update_target_every = target_extractor_parameters["ratio_update_target_every"]

        self._model_shape = [self._model_width] * self._model_num_layers

        self._ratio_model = simple_nets.MLP(input_size=self._model_state_size, hidden_sizes=self._model_shape, output_size=1)
        self._ratio_target_model = simple_nets.MLP(input_size=self._model_state_size, hidden_sizes=self._model_shape, output_size=1)
        self._behavior_policies = behavior_policies
        self._target_policies = target_policies 
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._ratio_model_lr)
        self._steps = 0

        self._target_mean = 0
        self._target_variance = 1
        self._target_count = 0
        

        self._update_target_network = self._create_target_network_update_op(self._ratio_model, self._ratio_target_model)

        self._state_ph = tf.placeholder(shape=[None, self._model_state_size], dtype=tf.float32, name="state_ph")
        self._next_state_ph = tf.placeholder(shape=[None, self._model_state_size], dtype=tf.float32, name="next_state_ph")
        self._action_probability_ratio_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="action_probability_ratio_ph")
        self._first_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="done_ph")
        self._target_mean_ph = tf.placeholder(shape=[1], dtype=tf.float32, name="target_mean_ph")
        self._target_std_ph = tf.placeholder(shape=[1], dtype=tf.float32, name="target_std_ph")
        self._accumulator_matrix_ph = tf.placeholder(shape=[None, None], dtype=tf.float32, name="accumulator_matrix_ph")

        # Computing the COP-TD discounted learning loss 
        state_ratios = tf.stop_gradient(self._ratio_target_model(self._state_ph))  # shape is [N, 1]
        next_state_ratios = self._ratio_model(self._next_state_ph)  # shape is [N, 1]
    

        # Compute the mean squared error of the loss function and save 
        self._target_value = self._ratio_model_discount * state_ratios * self._action_probability_ratio_ph + (1 - self._ratio_model_discount)  # shape is [N, 1]
        
        # Insert self._first_ph in computation so that we set first_states to the action_probability_ratio_ph 
        self._target_value = (1 - self._first_ph) * self._target_value + (self._first_ph) * self._action_probability_ratio_ph

        # self._target_value = (self._target_value - self._target_mean_ph) / self._target_std_ph

        # Mean squared error
        self._ratio_loss = tf.reduce_mean(tf.square(self._target_value - next_state_ratios))


        # Computing the Soft Ratio Normalization 
        self._state_ratios_soft = self._ratio_model(self._state_ph)  # shape is [N, 1]

        # Calculate the soft ratio normalization loss
        summed_ratios = tf.linalg.matmul(self._accumulator_matrix_ph, tf.stop_gradient(self._state_ratios_soft)) # shape is [N, 1]
        weighted_by_sum_ratios = (summed_ratios - 1) * self._state_ratios_soft # shape is [N, 1]
        self._soft_ratio_normalization_loss = tf.reduce_mean(weighted_by_sum_ratios)

        self._loss = self._ratio_loss + self._ratio_model_soft_normalization * self._soft_ratio_normalization_loss
        self._learn_step = self._optimizer.minimize(self._loss)

        self._initialize()

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

    def predict_ratio(self, states, numpy=False):
        """
        Given the behavior policies (of the dataset) and target policies (the policies currently being trained), compute the 
        state-occupancy ratios from the current ratio model.

        states: a numpy or tf array of size [N, self._model_state_size]

        Return: a tf array of size [N, 1] representing the occupancy ratios of each state
        """
        if not tf.is_tensor(states):
            states = tf.convert_to_tensor(states)

        ratio = tf.clip_by_value(self._ratio_model(states), 0, np.inf)

        if numpy:
            return tf.stop_gradient(ratio).eval() # Size [N, 1]
        else:
            return tf.stop_gradient(ratio)  # Size [N, 1]

    def train_ratio_model(self, number_of_training_steps):

        # Precompute all of the execution probabilities of OTHER players (using the "between" info_states, actions, players)
        probability_ratios = []
        global_states = []
        next_global_states = []
        firsts = []
        for player_data in self._data:
            for player in self._relevant_players:
                for trajectory in player_data:
                    for transition in trajectory:
                        # TODO: Insert handling for IS_FIRST so that the appropriate label is inserted for next_info_states

                        # Evaluation policies 
                        info_state = transition.info_state
                        legal_actions_mask = transition.legal_actions_mask 
                        action = transition.action 

                        # Get the action probability for player 
                        evaluation_probability = 1 if transition.is_first else self._target_policies[player].probabilities_with_actions([info_state], [[action]], [legal_actions_mask], numpy=True)[0][0]

                        # Get the product of action probabilities for each between_players
                        for curr_info_state, curr_legal_actions_mask, curr_action, curr_player in zip(transition.between_info_states, transition.between_legal_actions_masks, transition.between_actions, transition.between_players):
                            evaluation_probability *= self._target_policies[curr_player].probabilities_with_actions([curr_info_state], [[curr_action]], [curr_legal_actions_mask], numpy=True)[0][0]

                        # Repeat for behavior policies 
                        behavior_probability = 1 if transition.is_first else self._behavior_policies[player].probabilities_with_actions([info_state], [[action]], [legal_actions_mask], numpy=True)[0][0]

                        for curr_info_state, curr_legal_actions_mask, curr_action, curr_player in zip(transition.between_info_states, transition.between_legal_actions_masks, transition.between_actions, transition.between_players):
                            behavior_probability *= self._behavior_policies[curr_player].probabilities_with_actions([curr_info_state], [[curr_action]], [curr_legal_actions_mask], numpy=True)[0][0]

                        # divide the two probabilities and add to the probability_ratios 
                        probability_ratios.append([evaluation_probability / behavior_probability])
                        global_states.append(transition.global_state)
                        next_global_states.append(transition.next_global_state)
                        firsts.append([transition.is_first])

        # For a bunch of steps 
        data_size = len(global_states)
        training_loss = []

        # probs_for_firsts = (np.array(firsts) / np.sum(firsts)).flatten()
        # for _ in range(self._update_target_every):
        #     random_indices = np.random.choice(data_size, size=self._ratio_model_batch_size, p=probs_for_firsts)

        #     curr_states = [global_states[i] for i in random_indices]
        #     curr_next_states = [next_global_states[i] for i in random_indices]
        #     curr_prob_ratios = [probability_ratios[i] for i in random_indices]
        #     curr_firsts = [firsts[i] for i in random_indices]

        #     loss = self.execute_learn_step_ratio_model(curr_states, curr_prob_ratios, curr_next_states, curr_firsts)
        #     training_loss.append(loss)

        for _ in range(number_of_training_steps):
            random_indices = np.random.choice(data_size, size=self._ratio_model_batch_size)

            curr_states = [global_states[i] for i in random_indices]
            curr_next_states = [next_global_states[i] for i in random_indices]
            curr_prob_ratios = [probability_ratios[i] for i in random_indices]
            curr_firsts = [firsts[i] for i in random_indices]

            loss = self.execute_learn_step_ratio_model(curr_states, curr_prob_ratios, curr_next_states, curr_firsts)
            training_loss.append(loss)

        window_average = 100
        averaged_training_loss = [np.mean(training_loss[i:i+window_average])for i in range(len(training_loss) - window_average + 1)]
        logging.info("Ratio model training finished. ")
        plt.plot(averaged_training_loss)
        plt.savefig("batch_{}_lr_{}_update_every_{}_test.jpg".format(self._ratio_model_batch_size, self._ratio_model_lr, self._update_target_every))
        return 
    
    def execute_learn_step_ratio_model(self, state, probability_ratios, next_state, is_first):
        """
        Given a batch of Transition objects, execute one learning step. 

        batch: a list of Transition objects

        Return: loss value of the learn step
        """ 
        """
        info_states = [t.info_state for t in batch]
        global_states = [t.global_states for t in batch]
        actions = [t.action for t in batch]
        next_global_states = [t.next_global_states for t in batch]
        legal_actions_mask = [t.legal_actions_mask for t in batch]
        """

        # Create an [N, N] matrix where the diagonal elements are 0 while off-diagonals are 1.
        accumulator_matrix = (1.0 / (len(state) - 1)) * np.ones([len(state)] * 2) - np.diag([1] * len(state))


        # Pass into the self._session and request self._loss and self._learn_step
        loss, ratios, ratio_loss, normalization_loss, _, target_value = self._session.run([self._loss, self._state_ratios_soft, self._ratio_loss, self._soft_ratio_normalization_loss, self._learn_step, self._target_value],
            feed_dict={
                self._state_ph: state,
                self._next_state_ph: next_state,
                self._action_probability_ratio_ph: probability_ratios,
                self._first_ph: is_first,
                self._accumulator_matrix_ph: accumulator_matrix,
                self._target_mean_ph: [self._target_mean],
                self._target_std_ph: [self._target_variance ** .5],
            })

        self._steps += 1
        # self.update_target_mean_and_variance(target_value)

        # Update target network 
        if self._steps % self._update_target_every == 0:
            self._session.run(self._update_target_network)
            self._steps = 0
        # print("target Values: ", target_value)
        # print("Ratios: ", ratios, "Ratio loss: ", ratio_loss, "  normalization loss: ", normalization_loss)

        return loss
    
    def update_target_mean_and_variance(self, target_values):
        batch_mean = np.mean(target_values)
        batch_variance = np.var(target_values)
        batch_count = np.shape(target_values)[0]

        total_count = self._target_count + batch_count 

        self._target_mean = (self._target_mean * self._target_count + batch_mean * batch_count) / total_count 
        self._target_variance = (self._target_variance * self._target_count + batch_variance * batch_count) / total_count 
        self._target_count = total_count 
        print("Target mean and variance: ", self._target_mean, self._target_variance)
        return 


    