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
import time
import pickle

from open_spiel.python import simple_nets
from open_spiel.python.utils.replay_buffer import ReplayBuffer
from open_spiel.python.algorithms.psro_v2 import utils

from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.policy_wrapper import PolicyWrapper
from open_spiel.python.algorithms.offline_psro.utils.utils import Step


# Temporarily disable TF2 behavior until code is updated.
tf.disable_v2_behavior()

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state is_final_step legal_actions_mask")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9


class DQN(PolicyWrapper):
  """DQN Agent implementation in TensorFlow.

  See open_spiel/python/examples/breakthrough_dqn.py for an usage example.
  """

  def __init__(self,
               session,
               state_representation_size,
               num_actions,
               graph,
               reward_normalizer=None,
               start_frozen=False,
               double=True,
               symmetric=False,
               hidden_layers_sizes=[50, 50],
               replay_buffer_capacity=int(2e4),
               batch_size=128,
               replay_buffer_class=ReplayBuffer,
               learning_rate=3e-4,
               update_target_network_every=1000,
               learn_every=10,
               discount_factor=.99,
               min_buffer_size_to_learn=1000,
               epsilon_start=.8,
               epsilon_end=0.05,
               epsilon_decay_duration=int(1e5),
               optimizer_str="adam",
               loss_str="mse"):
    """Initialize the DQN agent."""

    # This call to locals() is used to store every argument used to initialize
    # the class instance, so it can be copied with no hyperparameter change.

    self._kwargs = locals()
    super().__init__(self, num_actions, state_representation_size) 

    ############# Parameter Initializations ############# 
    self._num_actions = num_actions
    if isinstance(hidden_layers_sizes, int):
      hidden_layers_sizes = [hidden_layers_sizes]
    self._layer_sizes = hidden_layers_sizes
    self._batch_size = batch_size
    self._update_target_network_every = update_target_network_every
    self._learn_every = learn_every
    self._min_buffer_size_to_learn = min_buffer_size_to_learn
    self._discount_factor = discount_factor
    self._reward_normalizer = reward_normalizer

    self._epsilon_start = epsilon_start
    self._epsilon_end = epsilon_end
    self._epsilon_decay_duration = epsilon_decay_duration
    self._double = double
    ########### Parameter Initializations End ###########

    # Buffer initializations
    if not isinstance(replay_buffer_capacity, int):
      raise ValueError("Replay buffer capacity not an integer.")
    self._replay_buffer = replay_buffer_class(replay_buffer_capacity)
    self._prev_step = None 
    self._prev_action = None

    # Step counter to keep track of learning, eps decay and target network.
    self._step_counter = 0
    self._num_updates = 0

    # Keep track of the last training loss achieved in an update step.
    self._last_loss_value = None
    self._loss_values_over_steps = []
    self._is_frozen = start_frozen
    self._graph = graph

    # Keep track of a flag to know whether we are optimizing for rewards or deviation coverage 
    self._optimizing_deviation_coverage = False
    self._num_trajectories = 0
    self._num_relabeled_non_halted_trajectories = 0
    self._num_relabeled_trajectories = 0
    
    # Cache Transitions so that we can modify rewards accordingly for deviation coverage optimizations 
    self._cache = []

    # Create required TensorFlow placeholders to perform the Q-network updates.
    self._info_state_ph = tf.placeholder(
        shape=[None, state_representation_size],
        dtype=tf.float32,
        name="info_state_ph")
    self._action_ph = tf.placeholder(
        shape=[None], dtype=tf.int32, name="action_ph")
    self._reward_ph = tf.placeholder(
        shape=[None], dtype=tf.float32, name="reward_ph")
    self._is_final_step_ph = tf.placeholder(
        shape=[None], dtype=tf.float32, name="is_final_step_ph")
    self._next_info_state_ph = tf.placeholder(
        shape=[None, state_representation_size],
        dtype=tf.float32,
        name="next_info_state_ph")
    self._legal_actions_mask_ph = tf.placeholder(
        shape=[None, num_actions],
        dtype=tf.float32,
        name="legal_actions_mask_ph")

    ############# Network Initializations ############# 
    self._q_network = simple_nets.MLP(state_representation_size,self._layer_sizes, num_actions)
    self._q_values = self._q_network(self._info_state_ph)
    self._target_q_network = simple_nets.MLP(state_representation_size,self._layer_sizes, num_actions)
    self._target_q_values = self._target_q_network(self._next_info_state_ph)
    ########### Network Initializations End ############

    # Target Network Update
    self._target_q_values = tf.stop_gradient(self._target_q_values)
    self._update_target_network = self._create_target_network_update_op(self._q_network, self._target_q_network)

    # Illegal Actions Mask
    illegal_actions = 1 - self._legal_actions_mask_ph
    illegal_logits = illegal_actions * ILLEGAL_ACTION_LOGITS_PENALTY

    ############# Bellman Update ############# 
    if self._double:
      next_q_double = self._q_network(self._next_info_state_ph)
      max_next_a = tf.math.argmax((tf.math.add(tf.stop_gradient(next_q_double), illegal_logits)), axis=-1)
      max_next_q = tf.gather(self._target_q_values, max_next_a, axis=1, batch_dims=1)
    else:
      max_next_q = tf.reduce_max(tf.math.add(tf.stop_gradient(self._target_q_values), illegal_logits),axis=-1)

    target = (self._reward_ph + (1 - self._is_final_step_ph) * self._discount_factor * max_next_q)
    action_indices = tf.stack([tf.range(tf.shape(self._q_values)[0]), self._action_ph], axis=-1)
    predictions = tf.gather_nd(self._q_values, action_indices)
    ########### Bellman Update End ###########

    self._savers = [("q_network", tf.train.Saver(self._q_network.variables)),("target_q_network", tf.train.Saver(self._target_q_network.variables))]

    if loss_str == "mse":
      loss_class = tf.losses.mean_squared_error
    elif loss_str == "huber":
      loss_class = tf.losses.huber_loss
    else:
      raise ValueError("Not implemented, choose from 'mse', 'huber'.")

    self._loss = tf.reduce_mean(
        loss_class(labels=target, predictions=predictions))

    if optimizer_str == "adam":
      self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer_str == "sgd":
      self._optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=learning_rate)
    else:
      raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")

    self._learn_step = self._optimizer.minimize(self._loss)
    if not self._is_frozen:
      self._initialize(session)

  def get_step_counter(self):
    return self._step_counter


  def _normalize(self, tensor, normalizer):
      return (tensor - normalizer.mean) / normalizer.standard_deviation 


  def step(self, step_object, player, session, is_evaluation=False):
    """Returns the action to be taken and updates the Q-network if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.
      add_transition_record: Whether to add to the replay buffer on this step.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    # Act step: don't act at terminal info states
  
    if not step_object.is_terminal:  # not terminal
        info_state = step_object.info_state 
        legal_actions_mask = step_object.legal_actions_mask
        legal_actions = [a for a, is_legal in enumerate(legal_actions_mask) if is_legal]
        epsilon = self._get_epsilon(is_evaluation)
        action, probs = self._epsilon_greedy(info_state, legal_actions, epsilon, session)
    else:
        action = None 
        probs = []

    # Don't mess with the state during evaluation.
    if not is_evaluation and not self._is_frozen:
      self._step_counter += 1

      if self._prev_step: 
        # Add to the cache 
        self.add_to_cache(self._prev_step, self._prev_action, step_object, player)

      if self._step_counter % self._learn_every == 0:
        self._last_loss_value = self.learn(session)

      if self._step_counter % self._update_target_network_every == 0:
        session.run(self._update_target_network)

      if step_object.is_terminal:
        # TODO: Modify this so that if it's terminal, we need to check if we are in deviation coverage optimization mode. If so, if HALTED, then modify all cache rewards to 0. Else, all to 1.
        self._num_trajectories += 1
        if self._optimizing_deviation_coverage: 
          self._num_relabeled_trajectories += 1
          self._num_relabeled_non_halted_trajectories += (1-int(step_object.halted))
          for cached_object in self._cache:
            step_to_modify = cached_object[2]
            if step_to_modify.is_terminal and (not step_to_modify.halted):
              new_reward = 1
            else:
              new_reward = 0
            cached_object[2] = Step(info_state=step_to_modify.info_state, reward=np.zeros(np.shape(step_to_modify.reward)) + new_reward, 
                                    is_terminal=step_to_modify.is_terminal, halted=step_to_modify.halted, acting_players=step_to_modify.acting_players, 
                                    global_state=step_to_modify.global_state, legal_actions_mask=step_to_modify.legal_actions_mask)
              
        # TODO: Then, transfer cache to buffer
        self.transfer_cache_to_buffer()

        self._prev_step = None 
        self._prev_action = None 
      else:
        self._prev_step = step_object
        self._prev_action = action 

    return action, probs

  def add_to_cache(self, prev_step, prev_action, step, player):
    self._cache.append([prev_step, prev_action, step, player])
    
  def transfer_cache_to_buffer(self):
    for item in self._cache:
      self.add_transition(*item)
    self._cache = []

  def set_deviation_coverage_flag(self, boolean_set):
    self._optimizing_deviation_coverage = boolean_set

  def add_transition(self, prev_step, prev_action, step, player):
    transition = Transition(
        info_state=prev_step.info_state,
        action=prev_action,
        reward=step.reward[0][player],
        next_info_state=step.info_state,
        is_final_step=float(step.is_terminal),
        legal_actions_mask=step.legal_actions_mask)
    self._replay_buffer.add(transition)

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
        tf.assign(target_v, v)
        for (target_v, v) in zip(self._target_variables, self._variables)
    ])

  def _epsilon_greedy(self, info_state, legal_actions, epsilon, session):
    """Returns a valid epsilon-greedy action and valid action probs.

    Action probabilities are given by a softmax over legal q-values.

    Args:
      info_state: hashable representation of the information state.
      legal_actions: list of legal actions at `info_state`.
      epsilon: float, probability of taking an exploratory action.

    Returns:
      A valid epsilon-greedy action and valid action probabilities.
    """
    probs = np.zeros(self._num_actions)

    if np.random.rand() < epsilon: # If this is a newly initialized policy, enforce that it is a uniform
      # action = np.random.choice(legal_actions)
      start = time.time()
      probs[legal_actions] = 1.0 / len(legal_actions)
      action = utils.random_choice(list(range(self._num_actions)), probs)
    else:
      info_state = np.reshape(info_state, [1, -1])
      start = time.time()
      if not self._is_frozen:
        q_values = session.run(
            self._q_values, feed_dict={self._info_state_ph: info_state})[0]            
      else:
        q_values = session.run(
            self._graph.get_tensor_by_name(self._frozen_output_variables["q_values"]), 
            feed_dict={self._graph.get_tensor_by_name(self._frozen_input_variables["info_state_ph"]): info_state})[0]                   
      legal_q_values = q_values[legal_actions]
      
      action = legal_actions[np.argmax(legal_q_values)]
      probs[action] = 1.0


    return action, probs

  def _get_epsilon(self, is_evaluation):
    """Returns the evaluation or decayed epsilon value."""
    if is_evaluation or self._is_frozen:
      return 0.0
    decay_steps = min(self._step_counter, self._epsilon_decay_duration)
    decayed_epsilon = (
        self._epsilon_end + (self._epsilon_start - self._epsilon_end) *
        (1 - decay_steps / self._epsilon_decay_duration))
    return decayed_epsilon

  def learn(self, session):
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """

    if (len(self._replay_buffer) < self._batch_size or
        len(self._replay_buffer) < self._min_buffer_size_to_learn):
      return None


    transitions = self._replay_buffer.sample(self._batch_size)
    info_states = [t.info_state for t in transitions]
    actions = [t.action for t in transitions]
    rewards = [self._normalize(t.reward, self._reward_normalizer) for t in transitions]
    next_info_states = [t.next_info_state for t in transitions]
    are_final_steps = [t.is_final_step for t in transitions]
    legal_actions_mask = [t.legal_actions_mask for t in transitions]

    loss, _ = session.run(
        [self._loss, self._learn_step],
        feed_dict={
            self._info_state_ph: info_states,
            self._action_ph: actions,
            self._reward_ph: rewards,
            self._is_final_step_ph: are_final_steps,
            self._next_info_state_ph: next_info_states,
            self._legal_actions_mask_ph: legal_actions_mask,
        })

    self._num_updates += 1

    return loss

  def _full_checkpoint_name(self, checkpoint_dir, name):
    checkpoint_filename = "_".join([name, "pid" + str(self.player_id)])
    return os.path.join(checkpoint_dir, checkpoint_filename)

  def _latest_checkpoint_filename(self, name):
    checkpoint_filename = "_".join([name, "pid" + str(self.player_id)])
    return checkpoint_filename + "_latest"

  @property
  def q_values(self):
    return self._q_values

  @property
  def replay_buffer(self):
    return self._replay_buffer

  @property
  def info_state_ph(self):
    return self._info_state_ph

  @property
  def loss(self):
    return self._last_loss_value

  @property
  def prev_timestep(self):
    return self._prev_timestep

  @property
  def prev_action(self):
    return self._prev_action

  @property
  def step_counter(self):
    return self._step_counter

  def get_output_variable_names(self):
      output_variables = [self._q_values]
      names = [var.name for var in output_variables]

      # self._frozen_output_variables should map to the tensors themselves, so we want the ':0'
      self._frozen_output_variables = {"q_values": self._q_values.name}
      self._frozen_input_variables = {"info_state_ph": self._info_state_ph.name}

      # For the names, we want only the operation. So, we get rid of the ':0' at the end
      names = [name[:name.index(":")] if ":" in name else name for i, name in enumerate(names)]
      return names

  def load_variable_names(self, variables):
      self._frozen_input_variables = variables["input"]
      self._frozen_output_variables = variables["output"]
      self._reward_normalizer = variables.get("reward_normalizer", None)

  def get_variable_name_file_name(self, policy_index):
      return "variable_names_policy_{}.pkl".format(policy_index)

  def freeze(self, model_manager, save_path):
      model_manager.freeze_graph(save_path, "policy_{}_frozen.pb".format(self._id), self.get_output_variable_names())
      frozen_graph = model_manager.load_frozen_graph(save_path, "policy_{}_frozen.pb".format(self._id))
      self._is_frozen = True 
      self._graph = frozen_graph
      self._replay_buffer.reset()
      save_variables = {"input": self._frozen_input_variables, "output": self._frozen_output_variables, "reward_normalizer": self._reward_normalizer}

      with open(save_path+self.get_variable_name_file_name(self._id), 'wb') as f:
        pickle.dump(save_variables, f)

      print("Policy is frozen. After {} gradient updates within {} environment steps.".format(self._num_updates, self._step_counter))
      print("Frozen model information:      Input variables: {}      Output variables: {} \n".format(self._frozen_input_variables, self._frozen_output_variables))
      # print("Operations: ", [op.name for op in self._frozen_graph.get_operations()])

  def get_graph(self):
      return self._graph

  def _initialize(self, session):
    initialization_weights = tf.group(
        *[var.initializer for var in self._variables])
    initialization_target_weights = tf.group(
        *[var.initializer for var in self._target_variables])
    initialization_opt = tf.group(
        *[var.initializer for var in self._optimizer.variables()])

    session.run(
        tf.group(*[
            initialization_weights, initialization_target_weights,
            initialization_opt,
        ]))

