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
import pickle

from open_spiel.python import simple_nets
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.world_model_base import WorldModelBase
from open_spiel.python.algorithms.offline_psro.utils.utils import get_terminal_state, compute_hash_string


class WorldModelDeterministic(WorldModelBase):
    frozen_graph_model_name = "dynamics_model_frozen.pb"
    model_arg_name = "model_args.pkl"

    def __init__(self, state_size, action_size, model_args={}, graph=None, start_frozen=False):
        super().__init__(self, state_size, action_size, model_args) 

        ############# Parameter Initialization #############
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
        self._true_state_extractor = model_args["true_state_extractor"]

        self._observation_output_size = sum(self._observation_sizes)
        self._legal_actions_output_size = self._num_actions if self._turn_based else self._num_actions * self._num_players

        # TODO: Get appropriate normalizers for everything?
        self._use_action_index = model_args["use_action_index"]
        self._true_state_normalizer = model_args["state_normalizer"]
        self._action_normalizer = model_args["action_normalizer"]

        if self._use_action_index:
            print("Using action index. Ensure that the action normalizer is 0 with standard deviation 1: ", self._action_normalizer)
        
        self._reward_normalizer = model_args["reward_normalizer"]
        self._observation_normalizer = model_args["observation_normalizer"]
        
        ########### Parameter Initialization End ###########
        if start_frozen:
            self._is_frozen = True 
            self._frozen_graph = graph
        else:
            ############# Model Initialization #############
            action_size = self._action_size * self._num_actions if self._use_action_index else self._action_size
            self._model_ensemble = [simple_nets.MLP(input_size=self._state_size+(action_size), hidden_sizes=self._hidden_sizes, output_size=self._state_size) for _ in range(self._ensemble_size)]
            self._reward_model = [simple_nets.MLP(input_size=self._state_size+(action_size), hidden_sizes=self._hidden_sizes, output_size=self._num_players) for _ in range(self._ensemble_size)]
            self._state_to_observation_and_legal_actions_mask = simple_nets.MLP(input_size=self._state_size, hidden_sizes=self._hidden_sizes, output_size=self._observation_output_size+self._legal_actions_output_size)
            ########### Model Initialization End ###########

            ############# Placeholder Initialization #############
            self._state_ph = tf.placeholder(shape=[None, self._state_size], dtype=tf.float32, name="state_ph")
            if self._use_action_index:
                self._action_ph = tf.placeholder(shape=[None, self._action_size], dtype=tf.int32, name="action_ph")
            else:
                self._action_ph = tf.placeholder(shape=[None, self._action_size], dtype=tf.float32, name="action_ph")
            self._next_state_ph = tf.placeholder(shape=[None, self._state_size], dtype=tf.float32, name="next_state_ph")
            self._reward_ph = tf.placeholder(shape=[None, self._num_players], dtype=tf.float32, name="reward_ph")
            self._observation_ph = tf.placeholder(shape=[None, sum(self._observation_sizes)], dtype=tf.float32, name="observation_ph")
            self._legal_actions_mask_ph = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32, name="legal_actions_mask_ph")
            ########### Placeholder Initialization End ###########

            ############# Forward Pass Calculations #############

            # Difference in state 
            self._delta_state = self._normalize(self._next_state_ph, self._true_state_normalizer) - self._normalize(self._state_ph, self._true_state_normalizer)  # [N, self._state_size]

            # Action one-hot encodings 

            if self._use_action_index:
                action_one_hot_vectors = tf.one_hot(self._action_ph, depth=self._num_actions, axis=-1)
                action_input = tf.concat([action_one_hot_vectors[:, i, :] for i in range(action_one_hot_vectors.shape[1])], axis=1)
            else:
                action_input = self._normalize(self._action_ph, self._action_normalizer)
            network_input = tf.concat([self._normalize(self._state_ph, self._true_state_normalizer), action_input], axis=1)  # [N, self._state_size + (self._action_size * self._num_actions)]

            # Pass through model ensemble
            self._model_output = []
            for i, model in enumerate(self._model_ensemble):
                curr_output = model(network_input)
                self._model_output.append(curr_output)

            self._reward_output = []
            for i, reward_model in enumerate(self._reward_model):
                self._reward_output.append(reward_model(network_input))

            # Concatenating 
            self._model_delta = tf.stack(self._model_output)  # [self._ensemble_size, N, self._state_size+self._num_players]

            # Reward estimation forward pass and contenation
            self._model_reward = tf.reduce_mean(self._reward_output, axis=0) # self._reward_model(network_input)  # [N, self._num_players]
            # self._model_reward = tf.concat([model(network_input) for model in self._reward_model], axis=2) # [self._ensemble_size, N, self._num_players]

            # Observation and Legal Actions Mappings
            observation_and_legal_actions_mask = self._state_to_observation_and_legal_actions_mask(self._normalize(self._state_ph, self._true_state_normalizer)) # [N, self._observation_output_size+self._legal_actions_output_size]

            self._observation = observation_and_legal_actions_mask[:, :self._observation_output_size]
            self._legal_actions_mask = observation_and_legal_actions_mask[:, -self._legal_actions_output_size:]
            
            # Loss Calculations
            self._dynamics_loss = 0
            for curr_output in self._model_output:
                self._dynamics_loss += tf.reduce_mean(tf.square(self._delta_state - curr_output))
            self._dynamics_loss /= self._ensemble_size
            self._reward_loss = 0
            for curr_output in self._reward_output:
                self._reward_loss += tf.reduce_mean(tf.square(self._normalize(self._reward_ph, self._reward_normalizer) - curr_output))
            self._reward_loss /= self._ensemble_size
            self._observation_loss = tf.reduce_mean(tf.square(self._observation - self._normalize(self._observation_ph, self._observation_normalizer)))
            self._legal_actions_loss = tf.reduce_mean(tf.square(self._legal_actions_mask_ph - self._legal_actions_mask))
            self._loss = self._dynamics_loss  + self._observation_loss + self._legal_actions_loss

            self._reward_output = tf.stack(self._reward_output)
            ########### Forward Pass Calculations End ###########

            ############# Final Initializations #############
            # Optimizers
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
            self._train_step = self._optimizer.minimize(self._loss)

            # Reward optimizer separate triaining
            self._reward_optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
            self._train_reward_step = self._reward_optimizer.minimize(self._reward_loss) 

            # Frozen stuff
            self._is_frozen = False
            self._frozen_graph = None

            self._initialize()
            ########### Final Initializations End ############
    
    def _normalize(self, tensor, normalizer):
        return (tensor - normalizer.mean) / normalizer.standard_deviation 

    def _unnormalize(self, tensor, normalizer):
        return (tensor * normalizer.standard_deviation) + normalizer.mean 

    def _initialize(self):
        # Initialize policy network, both value networks, all three target networks, and both optimizers
        variable_list = []
        for model in self._model_ensemble:
            init_model = tf.group(*[var.initializer for var in model.variables[:]])
            variable_list.append(init_model)
        init_opt = tf.group(*[var.initializer for var in self._optimizer.variables()])
        init_reward_opt = tf.group(*[var.initializer for var in self._reward_optimizer.variables()])
        variable_list.append(init_opt)
        variable_list.append(init_reward_opt)
        # variable_list.append(tf.group(*[var.initializer for var in self._state_to_observation.variables[:]]))
        # variable_list.append(tf.group(*[var.initializer for var in self._state_to_legal_actions.variables[:]]))
        variable_list.append(tf.group(*[var.initializer for var in self._state_to_observation_and_legal_actions_mask.variables[:]]))

        for model in self._reward_model:
            variable_list.append(tf.group(*[var.initializer for var in model.variables[:]]))

        # variable_list.append(tf.group(*[var.initializer for var in self._reward_model.variables[:]]))

        self._session.run(tf.group(*variable_list))
        return       

    def get_output_variable_names(self):
        ############# Set Frozen Variable Names #############
        output_variables = [self._model_delta, self._model_reward, self._reward_output, self._observation, self._legal_actions_mask]
        names = [var.name for var in output_variables]

        # self._frozen_output_variables should map to the tensors themselves, so we want the ':0'
        self._frozen_output_variables = {"delta": names[0], "reward": names[1], "reward_ensemble": names[2], "observation": names[3], "legal_actions_mask": names[4]}
        self._frozen_input_variables = {"state_ph": self._state_ph.name, "action_ph": self._action_ph.name} # "next_state_ph": self._next_state_ph.name}

        # For the names, we want only the operation. So, we get rid of the ':0' at the end
        names = [name[:name.index(":")] if ":" in name else name for i, name in enumerate(names)]
        ########### Set Frozen Variable Names End ###########
        return names
        
    def load_variable_names(self, variables):
        self._frozen_input_variables = variables["input"]
        self._frozen_output_variables = variables["output"]
        self._true_state_normalizer = variables["true_state_normalizer"]
        self._action_normalizer = variables["action_normalizer"]
        self._reward_normalizer = variables["reward_normalizer"]
        self._observation_normalizer = variables["observation_normalizer"]

    def get_variable_name_file_name(self):
        return "variable_names_model.pkl"

    def freeze(self, model_manager, save_path):
        # Freeze the dynamics model and save the frozen file
        model_manager.freeze_graph(save_path, WorldModelDeterministic.frozen_graph_model_name, self.get_output_variable_names())
        frozen_graph = model_manager.load_frozen_graph(save_path, WorldModelDeterministic.frozen_graph_model_name)
        self._is_frozen = True 
        self._frozen_graph = frozen_graph

        save_variables = {"input": self._frozen_input_variables, "output": self._frozen_output_variables, 
                        "true_state_normalizer": self._true_state_normalizer, "action_normalizer": self._action_normalizer, 
                        "reward_normalizer": self._reward_normalizer, "observation_normalizer": self._observation_normalizer}

        with open(save_path+self.get_variable_name_file_name(), 'wb') as f:
            pickle.dump(save_variables, f)
        print("Dynamics model is frozen.")
        print("Dynamics model info:     Input variables: {}      Output variables: {} \n".format(self._frozen_input_variables, self._frozen_output_variables))
        # print("Operations: ", [op.name for op in self._frozen_graph.get_operations()])

    def get_frozen_graph(self):
        return self._frozen_graph

    def get_next_step(self, state, action, frozen_session=None):
        # Repeat tensors to pass into the network ensemble
        state = np.expand_dims(state, axis=0)

        
        # If not using indices, convert action to semantic representation using true_state_extractor 
        if not self._use_action_index:
            action = [self._true_state_extractor.action_index_to_vector_representation(a) for a in action]
        else:
            action = np.expand_dims(action, axis=0)


        # Pass state and action through computational graphs (frozen or not)
        if not self._is_frozen:
            deltas, reward, reward_ensemble = self._session.run(
                [self._model_delta, self._model_reward, self._reward_output], 
                feed_dict={
                    self._state_ph: state,
                    self._action_ph: action,
                }
            )
        else:
            state_input = self._frozen_graph.get_tensor_by_name(self._frozen_input_variables["state_ph"])
            action_input = self._frozen_graph.get_tensor_by_name(self._frozen_input_variables["action_ph"])
            delta_output = self._frozen_graph.get_tensor_by_name(self._frozen_output_variables['delta'])
            model_reward = self._frozen_graph.get_tensor_by_name(self._frozen_output_variables['reward'])
            reward_ensemble_output = self._frozen_graph.get_tensor_by_name(self._frozen_output_variables['reward_ensemble'])
            deltas, reward, reward_ensemble = frozen_session.run([delta_output, model_reward, reward_ensemble_output], feed_dict={state_input: state, action_input: action})

        # State deltas added back into original states

        # Unnormalize all of the outputs:
        # deltas = [self._unnormalize(deltas[i], self._true_state_normalizer) for i in range(self._ensemble_size)]
        reward = self._unnormalize(reward, self._reward_normalizer)
        reward_ensemble = np.array([self._unnormalize(reward_ensemble[i], self._reward_normalizer) for i in range(self._ensemble_size)])

        predictions = [self._unnormalize(self._normalize(state, self._true_state_normalizer) + deltas[i], self._true_state_normalizer) for i in range(self._ensemble_size)]

        # Predict the next state by averaging over predictions in the ensemble 
        if self._indicator_rounding:
            aggregate_next_state = np.rint(np.mean(predictions, axis=0))
        else:
            aggregate_next_state = np.mean(predictions, axis=0)

        # Predict the next reward by averaging over predictions in the ensemble and renormalizing accordingly
        # aggregate_reward = np.mean(reward, axis=0)
        
        aggregate_reward = reward #  * self._reward_span) + self._reward_min
        # reward_ensemble = [(r * self._reward_span) + self._reward_min for r in reward_ensemble]
        
        player_observations, legal_actions_masks = self.get_observations(aggregate_next_state, frozen_session)

        # Check if the model predicts a terminal state
        term_state = get_terminal_state(1, self._state_size)
        is_terminal = np.mean(np.abs(aggregate_next_state - term_state)) < .5

        # If it is terminal, there are no legal actions to do next
        if is_terminal:
            legal_actions_masks = np.array([0.0 for _ in range(self._num_actions if self._turn_based else self._num_actions * self._num_players)])
        # If for some reason our model says it's not terminal but we have no legal actions, choose a random one and say it's valid. 
        elif np.sum(legal_actions_masks) < 1: 
            legal_actions_masks = np.array([1.0 for _ in range(self._num_actions if self._turn_based else self._num_actions * self._num_players)])

        # If we provide information that the game only has terminal rewards, then set rewards to 0 if not terminal
        if self._reward_structure == "terminal" and not is_terminal:
            aggregate_reward = np.zeros(aggregate_reward.shape)
            max_reward_discrepancy = 0
            reward_ensemble = np.zeros(reward_ensemble.shape)
        else:
            # MAX DISCREPANCY ACROSS ENSEMBLE
            max_reward_discrepancy = -np.inf

            for i in range(self._ensemble_size):
                metric = np.mean(np.abs(aggregate_reward - reward_ensemble[i]))
                max_reward_discrepancy = max(max_reward_discrepancy, metric)

        max_prediction_discrepancy = max_reward_discrepancy

        aggregate_next_state = np.squeeze(aggregate_next_state)
        reward_ensemble = np.swapaxes(np.squeeze(reward_ensemble), 0, 1)
        return aggregate_next_state, aggregate_reward, player_observations, legal_actions_masks, is_terminal, max_prediction_discrepancy, reward_ensemble
    
    def get_observations(self, state, frozen_session=None):
        # Retrieve the observations and legal action masks given the current state. Output size depends on turn-based or simultaneous
        # state = np.expand_dims(state, axis=0)
        # state = np.repeat(state, self._ensemble_size, axis=0)
        
        if not self._is_frozen:
            observations, legal_actions_masks = self._session.run(
                [self._observation, self._legal_actions_mask],
                feed_dict={
                    self._state_ph: state
                }
            )
        else:
            state_input = self._frozen_graph.get_tensor_by_name(self._frozen_input_variables["state_ph"])
            observation_output = self._frozen_graph.get_tensor_by_name(self._frozen_output_variables["observation"])
            legal_actions_mask_output = self._frozen_graph.get_tensor_by_name(self._frozen_output_variables["legal_actions_mask"])
            observations, legal_actions_masks = frozen_session.run(
                [observation_output, legal_actions_mask_output],
                feed_dict={
                    state_input: state
                }
            )

        index = 0
        player_observations = []
        # observations = np.mean(observations, axis=0)
        for size in self._observation_sizes:
            if self._indicator_rounding:
                player_observations.append(np.rint(np.squeeze(observations[:, index:index+size])))
            else:
                player_observations.append(self._unnormalize(np.squeeze(observations[:, index:index+size]), self._observation_normalizer))
            index += size

        # TODO: If this is a simultaneous game, we would need to split the output of the self._state_to_legal_actions for corresponding players
        legal_actions_masks = np.squeeze(legal_actions_masks, axis=0)
        legal_actions_masks = np.clip(np.rint(legal_actions_masks), 0, 1)
        
        return player_observations, legal_actions_masks
    
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
        ########################## Preprocessing End ############################

        ######################### Training Loop #################################
        for _ in range(num_gradient_steps):
            states, actions, next_states, observations, rewards, legal_actions_mask = [], [], [], [], [], []

            # Keeping sampling for each model separate to lessen correlation
            # for ensemble_index in range(self._ensemble_size):  
                
                ######### Batch sampling for model number ensemble_index #########
            indices = np.random.choice(data_size, self._batch_size)
            # states.append([])
            # actions.append([])
            # next_states.append([])
            # observations.append([])
            # legal_actions_mask.append([])
            # rewards.append([])

            # Append each of the sampled datapoints to the current model (ensemble_index-th model)
            for i in indices:
                states.append(self._data[i].global_state)
                if self._use_action_index:
                    actions.append([self._data[i].actions[p] for p in self._data[i].relevant_players])  
                else:
                    actions.append(np.concatenate([self._true_state_extractor.action_index_to_vector_representation(self._data[i].actions[p]) for p in self._data[i].relevant_players], axis=0))
                legal_actions_mask.append(np.hstack([self._data[i].legal_actions_masks[p] for p in self._data[i].relevant_players]))
                rewards.append(self._data[i].rewards)#  - self._reward_min) / self._reward_span)

                # If the next state terminal (None), get the manual terminal vector from utils.get_terminal_state
                next_states.append(self._data[i].next_global_state if self._data[i].next_global_state != None else get_terminal_state(1, self._state_size)[0])
                
                # Make the mapping from the current state to current observations
                observations.append(np.hstack(self._data[i].info_states))

                # TODO: Acting next acting players observations

            # Learning step
            loss, dynamics_loss, observation_loss, mask_loss, _ = self._session.run(
                [self._loss, self._dynamics_loss, self._observation_loss, self._legal_actions_loss, self._train_step], 
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

        #################### Reward Training Loop #############################
        all_reward_losses = []
        if self._reward_structure == "terminal":
            data_indices = [i for i, d in enumerate(self._data) if d.done]
            print("Condensed reward dataset from {} datapoints to {} datapoints".format(len(self._data), len(data_indices)))

            count_greater_0 = 0
            for j in data_indices:
                curr_t = self._data[j]
                if np.sum(curr_t.rewards) > 0:
                    count_greater_0 += 1
            print("Proportion with greater than 0 reward: ", float(count_greater_0) / len(data_indices))
            self._reward_normalizer.modify(mean=0, standard_deviation=np.std([self._data[i].rewards for i in data_indices]))
            print("Recalculating normalizer based on terminal rewards: ", self._reward_normalizer)
        else:
            data_indices = [i for i in range(len(self._data))]

        for _ in range(num_gradient_steps):
            states, actions, rewards = [], [], []

            # Keeping sampling for each model separate to lessen correlation
            # for ensemble_index in range(self._ensemble_size):  
            
            ######### Batch sampling for model number ensemble_index #########
            indices = np.random.choice(data_indices, self._batch_size)
            # states.append([])
            # actions.append([])
            # rewards.append([])

            # Append each of the sampled datapoints to the current model (ensemble_index-th model)
            for i in indices:
                states.append(self._data[i].global_state)
                if self._use_action_index:
                    actions.append([self._data[i].actions[p] for p in self._data[i].relevant_players])  
                else:
                    actions.append(np.concatenate([self._true_state_extractor.action_index_to_vector_representation(self._data[i].actions[p]) for p in self._data[i].relevant_players], axis=0))
                rewards.append(self._data[i].rewards) #  - self._reward_min) / self._reward_span)

            # Learning step
            reward_loss, _, reward_output = self._session.run(
                [self._reward_loss, self._train_reward_step, self._reward_output], 
                feed_dict={
                    self._state_ph: states,
                    self._action_ph: actions,
                    self._reward_ph: rewards,
                }
            )
            # print("reward loss: ", reward_loss)
            # print("original rewards: ", np.array(rewards))
            # print("Targets: ", self._normalize(np.array(rewards), self._reward_normalizer), "    Outputs: ", reward_output)
            # # print("Max dev: ", np.max(np.mean(np.abs(reward_output - np.expand_dims(np.mean(reward_output, axis=0), axis=0)), axis=1)))
            # print("Average variance across reward ensemble: ", np.mean(np.square(reward_output - np.expand_dims(np.mean(reward_output, axis=0), axis=0))))
            all_reward_losses.append(reward_loss)

        #################### Reward Training End #############################
        return all_losses, all_reward_losses


