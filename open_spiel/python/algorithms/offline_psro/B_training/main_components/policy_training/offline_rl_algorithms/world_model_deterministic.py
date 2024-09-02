"""
This file is dedicated to training a feed-forward world model
"""

import collections
import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import math 
import time

from open_spiel.python import simple_nets
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.world_model_base import WorldModelBase
from open_spiel.python.algorithms.offline_psro.utils.utils import get_terminal_state, compute_hash_string

class WorldModelDeterministic(WorldModelBase):
    def __init__(self, state_size, action_size, model_args={}):
        super().__init__(self, state_size, action_size, model_args) 

        self._hidden_sizes = model_args["hidden_sizes"]
        self._batch_size = model_args["batch_size"]
        self._lr = model_args["learning_rate"]
        self._ensemble_size = model_args["ensemble_size"]
        self._observation_sizes = model_args["observation_sizes"]
        self._num_players = model_args["num_players"]
        self._num_actions = model_args["num_actions"] # This is used to normalize action inputs between -1 and 1
        self._turn_based = model_args["turn_based"]
        self._indicator_rounding = model_args["indicator_rounding"]
        self._reward_structure = model_args["reward_structure"]

        self._model_ensemble = [simple_nets.MLP(input_size=self._state_size+(self._action_size * self._num_actions), hidden_sizes=self._hidden_sizes, output_size=self._state_size) for _ in range(self._ensemble_size)]
        self._reward_model = [simple_nets.MLP(input_size=self._state_size+(self._action_size * self._num_actions), hidden_sizes=self._hidden_sizes, output_size=1) for _ in range(self._num_players)]
        self._state_to_observation = simple_nets.MLP(input_size=self._state_size, hidden_sizes=self._hidden_sizes, output_size=sum(self._observation_sizes))
        self._state_to_legal_actions = simple_nets.MLP(input_size=self._state_size, hidden_sizes=self._hidden_sizes, output_size=self._num_actions if self._turn_based else self._num_actions * self._num_players)

        self._state_ph = tf.placeholder(shape=[self._ensemble_size, None, self._state_size], dtype=tf.float32, name="state_ph")
        self._action_ph = tf.placeholder(shape=[self._ensemble_size, None, self._action_size], dtype=tf.int32, name="action_ph")
        self._next_state_ph = tf.placeholder(shape=[self._ensemble_size, None, self._state_size], dtype=tf.float32, name="next_state_ph")
        self._reward_ph = tf.placeholder(shape=[self._ensemble_size, None, self._num_players], dtype=tf.float32, name="reward_ph")
        self._observation_ph = tf.placeholder(shape=[self._ensemble_size, None, sum(self._observation_sizes)], dtype=tf.float32, name="observation_ph")
        self._legal_actions_mask_ph = tf.placeholder(shape=[self._ensemble_size, None, self._num_actions], dtype=tf.float32, name="legal_actions_mask_ph")
        
        self._delta_state = self._next_state_ph - self._state_ph  # [self._ensemble_size, N, self._state_size]

        action_one_hot_vectors = tf.one_hot(self._action_ph, depth=self._num_actions, axis=-1)
        action_one_hot_vectors = tf.concat([action_one_hot_vectors[:, :, i, :] for i in range(action_one_hot_vectors.shape[2])], axis=2)
        network_input = tf.concat([self._state_ph, action_one_hot_vectors], axis=2)  # [self._ensemble_size, N, self._state_size + (self._action_size * self._num_actions)]

        self._model_output = []
        for i, model in enumerate(self._model_ensemble):
            curr_output = model(network_input[i])
            self._model_output.append(curr_output)

        self._model_output = tf.stack(self._model_output)  # [self._ensemble_size, N, self._state_size+self._num_players]
        self._model_delta = self._model_output# [:, :, :-self._num_players]

        self._model_reward = tf.concat([model(network_input) for model in self._reward_model], axis=2) # [self._ensemble_size, N, self._num_players]

        self._dynamics_loss = tf.reduce_mean(tf.square(self._delta_state - self._model_delta))
        self._reward_loss = tf.reduce_mean(tf.square(self._reward_ph - self._model_reward))

        # Observation mapping 
        self._observation = self._state_to_observation(self._next_state_ph)
        self._observation_loss = tf.reduce_mean(tf.square(self._observation - self._observation_ph))

        # Legal Actions mapping
        self._legal_actions_mask = self._state_to_legal_actions(self._state_ph)
        self._legal_actions_loss = tf.reduce_mean(tf.square(self._legal_actions_mask_ph - self._legal_actions_mask))

        self._loss = self._dynamics_loss + self._reward_loss + self._observation_loss + self._legal_actions_loss 

        # Optimizers
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
        self._train_step = self._optimizer.minimize(self._loss)

        # Frozen stuff
        self._is_frozen = False
        self._frozen_graph = None

        self._initialize()


    def _initialize(self):
        # Initialize policy network, both value networks, all three target networks, and both optimizers
        variable_list = []
        for model in self._model_ensemble:
            init_model = tf.group(*[var.initializer for var in model.variables[:]])
            variable_list.append(init_model)
        init_opt = tf.group(*[var.initializer for var in self._optimizer.variables()])
        variable_list.append(init_opt)
        variable_list.append(tf.group(*[var.initializer for var in self._state_to_observation.variables[:]]))
        variable_list.append(tf.group(*[var.initializer for var in self._state_to_legal_actions.variables[:]]))

        for model in self._reward_model:
            variable_list.append(tf.group(*[var.initializer for var in model.variables[:]]))

        self._session.run(tf.group(*variable_list))
        return       

    def get_output_variable_names(self):
        output_variables = [self._model_delta, self._model_reward, self._observation, self._legal_actions_mask]
        names = [var.name for var in output_variables]

        # self._frozen_output_variables should map to the tensors themselves, so we want the ':0'
        self._frozen_output_variables = {"delta": names[0], "reward": names[1], "observation": names[2], "legal_actions_mask": names[3]}
        self._frozen_input_variables = {"state_ph": self._state_ph.name, "action_ph": self._action_ph.name, "next_state_ph": self._next_state_ph.name}

        # For the names, we want only the operation. So, we get rid of the ':0' at the end
        names = [name[:name.index(":")] if ":" in name else name for i, name in enumerate(names)]
        return names

    def freeze(self, model_manager, save_path):
        model_manager.freeze_graph(save_path, "dynamics_model_frozen.pb", self.get_output_variable_names())
        frozen_graph = model_manager.load_frozen_graph(save_path, "dynamics_model_frozen.pb")
        self._is_frozen = True 
        self._frozen_graph = frozen_graph
        print("Dynamics model is frozen.")
        print("Dynamics model info:     Input variables: {}      Output variables: {} \n".format(self._frozen_input_variables, self._frozen_output_variables))
        # print("Operations: ", [op.name for op in self._frozen_graph.get_operations()])

    def get_frozen_graph(self):
        return self._frozen_graph

    def get_next_step(self, state, action, halt_threshold, frozen_session=None):
        # Repeat tensors to pass into the network ensemble
        state = np.expand_dims(np.expand_dims(state, axis=0), axis=0)
        action = np.expand_dims(np.expand_dims(action, axis=0), axis=0)

        state = np.repeat(state, self._ensemble_size, axis=0)
        action = np.repeat(action, self._ensemble_size, axis=0)

        # Pass state and action through computational graphs (frozen or not)
        if not self._is_frozen:
            deltas, reward = self._session.run(
                [self._model_delta, self._model_reward], 
                feed_dict={
                    self._state_ph: state,
                    self._action_ph: action,
                }
            )
        else:
            state_input = self._frozen_graph.get_tensor_by_name(self._frozen_input_variables["state_ph"])
            action_input = self._frozen_graph.get_tensor_by_name(self._frozen_input_variables["action_ph"])
            delta_output = self._frozen_graph.get_tensor_by_name(self._frozen_output_variables['delta'])
            reward_output = self._frozen_graph.get_tensor_by_name(self._frozen_output_variables['reward'])
            deltas, reward = frozen_session.run([delta_output, reward_output], feed_dict={state_input: state, action_input: action})

        # State deltas added back into original states
        predictions = [state[i] + deltas[i] for i in range(self._ensemble_size)]

        # Apply indicator rounding if applicable 
        if self._indicator_rounding:
            predictions = [np.rint(pred) for pred in predictions]

        # Check whether we've reached a HALT-state. In indicator settings, we check whether at least 1 bit has changed.
        # Otherwise, we use the MSE and a self._halt_threshold to determine the stop condition
        halt = False 
        for i in range(self._ensemble_size):
            for j in range(self._ensemble_size):
                if self._indicator_rounding:
                    metric = np.max(np.abs(predictions[i] - predictions[j])) # tv
                else:
                    metric = np.mean(np.square(predictions[i] - predictions[j])) # mse
                
                if metric >= halt_threshold:
                    halt = True
                    break 

        # Predict the next state by averaging over predictions in the ensemble 
        aggregate_next_state = np.mean(predictions, axis=0)

        # Predict the next reward by averaging over predictions in the ensemble and renormalizing accordingly
        aggregate_reward = np.mean(reward, axis=0)
        aggregate_reward = (aggregate_reward * self._reward_span) + self._reward_min
        
        # Get the subsequent observations and legal_action_masks given the next state prediction for the policies
        next_state_repeated_input = np.repeat(np.expand_dims(aggregate_next_state, axis=0), self._ensemble_size, axis=0)
        if not self._is_frozen:
            observations, legal_actions_masks = self._session.run(
                [self._observation, self._legal_actions_mask],
                feed_dict={
                    self._next_state_ph: next_state_repeated_input,
                    self._state_ph: next_state_repeated_input
                }
            )
        else:
            next_state_input = self._frozen_graph.get_tensor_by_name(self._frozen_input_variables["next_state_ph"])
            state_input = self._frozen_graph.get_tensor_by_name(self._frozen_input_variables["state_ph"])
            observation_output = self._frozen_graph.get_tensor_by_name(self._frozen_output_variables["observation"])
            legal_actions_mask_output = self._frozen_graph.get_tensor_by_name(self._frozen_output_variables["legal_actions_mask"])
            observations, legal_actions_masks = frozen_session.run(
                [observation_output, legal_actions_mask_output],
                feed_dict={
                    next_state_input: next_state_repeated_input,
                    state_input: next_state_repeated_input
                }
            )
        
        # TODO: If this is a simultaneous game, we would need to split the output of the self._state_to_legal_actions for corresponding players
        legal_actions_masks = np.squeeze(np.mean(legal_actions_masks, axis=0))
        legal_actions_masks = np.clip(np.rint(legal_actions_masks), 0, 1)

        # Split observations based on player 
        index = 0
        player_observations = []
        observations = np.mean(observations, axis=0)
        for size in self._observation_sizes:
            if self._indicator_rounding:
                player_observations.append(np.rint(np.squeeze(observations[:, index:index+size])))
            else:
                player_observations.append(np.squeeze(observations[:, index:index+size]))
            index += size

        # If we have reached a halt, return 0 reward, no legal actions, and terminal flag
        if halt:
            reward = [[0 for _ in range(self._num_players)]]  # Any penalty (MoREL) handling is done outside of this class
            legal_actions_masks = [0 for _ in range(self._num_actions if self._turn_based else self._num_actions * self._num_players)]
            return aggregate_next_state, reward, player_observations, legal_actions_masks, 1, halt

        # Check if we did not reach a halt but, instead, the model predicts a terminal state
        term_state = get_terminal_state(1, self._state_size)
        is_terminal = np.mean(aggregate_next_state - term_state) < 5e-2

        # If it is terminal, there are no legal actions to do next
        if is_terminal:
            legal_actions_masks = np.array([0.0 for _ in range(self._num_actions if self._turn_based else self._num_actions * self._num_players)])
        # If for some reason our model says it's not terminal but we have no legal actions, choose a random one and say it's valid. 
        elif np.sum(legal_actions_masks) < 1: 
            legal_actions_masks = np.array([1.0 for _ in range(self._num_actions if self._turn_based else self._num_actions * self._num_players)])
        
        # If we provide information that the game only has terminal rewards, then set rewards to 0 if not terminal
        if self._reward_structure == "terminal" and not is_terminal:
            aggregate_reward = np.zeros(aggregate_reward.shape)

        aggregate_next_state = np.squeeze(aggregate_next_state)
        return aggregate_next_state, aggregate_reward, player_observations, legal_actions_masks, is_terminal, halt
    
    def train(self, num_gradient_steps):
        state_action_to_reward = {}
        repeated = []
        for i, t in enumerate(self._data):
            true_state = t.global_state
            action = t.actions[t.relevant_players[0]]

            hash_string = compute_hash_string(true_state)
            hash_string = hash_string + str(action)
    
            if hash_string in state_action_to_reward:
                curr_reward = state_action_to_reward[hash_string]
    
                if not all([t.rewards[i] == curr_reward[i] for i in range(len(curr_reward))]):
                    print("repeated!: ",  curr_reward, t.rewards, hash_string)
                    repeated.append(i)
            else:
                state_action_to_reward[hash_string] = t.rewards
        
        if len(repeated) > 0:
            print("Warning: training a deterministic model on a non-deterministic dataset. Found {} repeats.".format(len(repeated)))

        self._data = [t for i, t in enumerate(self._data) if i not in repeated]
        ############################ Preprocessing ##############################
        data_size = len(self._data)
        all_losses = []

        self._reward_min = np.min([d.rewards for d in self._data], axis=0)
        self._reward_max = np.max([d.rewards for d in self._data], axis=0)
        self._reward_span = self._reward_max - self._reward_min
        ########################## Preprocessing End ############################

        ######################### Training Loop #################################
        for _ in range(num_gradient_steps):
            states, actions, next_states, observations, rewards, legal_actions_mask = [], [], [], [], [], []

            # Keeping sampling for each model separate to lessen correlation
            for ensemble_index in range(self._ensemble_size):  
                
                ######### Batch sampling for model number ensemble_index #########
                indices = np.random.choice(data_size, self._batch_size)
                states.append([])
                actions.append([])
                next_states.append([])
                observations.append([])
                legal_actions_mask.append([])
                rewards.append([])

                # Append each of the sampled datapoints to the current model (ensemble_index-th model)
                for i in indices:
                    states[ensemble_index].append(self._data[i].global_state)
                    actions[ensemble_index].append([self._data[i].actions[p] for p in self._data[i].relevant_players])  
                    legal_actions_mask[ensemble_index].append(np.hstack([self._data[i].legal_actions_masks[p] for p in self._data[i].relevant_players]))
                    rewards[ensemble_index].append((self._data[i].rewards - self._reward_min) / self._reward_span)

                    # If the next state terminal (None), get the manual terminal vector from utils.get_terminal_state
                    next_states[ensemble_index].append(self._data[i].next_global_state if self._data[i].next_global_state != None else get_terminal_state(1, self._state_size)[0])
                    
                    # If the next state is terminal, get the manual terminal vector from utils.get_terminal_state
                    observations[ensemble_index].append(get_terminal_state(1, sum(self._observation_sizes))[0] if all([o == None for o in self._data[i].next_info_states]) else np.hstack(self._data[i].next_info_states))

                    # TODO: Acting next acting players observations

            # Learning step
            loss, dynamics_loss, reward_loss, observation_loss, mask_loss, _ = self._session.run(
                [self._loss, self._dynamics_loss, self._reward_loss, self._observation_loss, self._legal_actions_loss, self._train_step], 
                feed_dict={
                    self._state_ph: states,
                    self._action_ph: actions,
                    self._next_state_ph: next_states,
                    self._reward_ph: rewards,
                    self._observation_ph: observations,
                    self._legal_actions_mask_ph: legal_actions_mask
                }
            )
            # Loss Tracking
            all_losses.append(loss)

        ####################### Training Loop End ###############################
        return all_losses


