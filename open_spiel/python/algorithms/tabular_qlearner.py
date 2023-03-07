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

"""Tabular Q-learning agent."""

import collections
import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state is_final_step legal_actions")

def valuedict():
  return collections.defaultdict(float)


class QLearner(rl_agent.AbstractAgent):
  """Tabular Q-Learning agent.

  See open_spiel/python/examples/tic_tac_toe_qlearner.py for an usage example.
  """

  def __init__(self,
               player_id,
               num_actions,
               state_representation_size,
               step_size=0.1,
               epsilon_schedule=rl_tools.LinearSchedule(1, .1, 1500000),
               discount_factor=.99,
               centralized=False):
    """Initialize the Q-Learning agent."""
    self._player_id = player_id
    self._num_actions = num_actions
    self._state_representation_size = state_representation_size  # not currently used
    self._step_size = step_size
    self._epsilon_schedule = epsilon_schedule
    self._epsilon = epsilon_schedule.value
    self._discount_factor = discount_factor
    self._centralized = centralized
    self._q_values = collections.defaultdict(valuedict)
    self._prev_info_state = None
    self._prev_action = None
    self._last_loss_value = None

    self._loss_values_over_steps = []
    self._total_steps = 0

    self.buffer = []
    self.boltzmann = 1000
    # self.episode_counter = 0
    # self.train_every_episode = 1
    # self.batch_size = 80
    # self.num_transitions_bake = 2000

    # self.buffer_size = 1000

    # self.no_buffer = True

  def _epsilon_greedy(self, info_state, legal_actions, epsilon, is_evaluation):
    """Returns a valid epsilon-greedy action and valid action probs.

    If the agent has not been to `info_state`, a valid random action is chosen.

    Args:
      info_state: hashable representation of the information state.
      legal_actions: list of actions at `info_state`.
      epsilon: float, prob of taking an exploratory action.

    Returns:
      A valid epsilon-greedy action and valid action probabilities.
    """
    probs = np.zeros(self._num_actions)
    # TODO: Consideration to make the epsilon greedy probabilistic when we are not in evaluation. This should help exploration a tiny bit
    """if self.no_buffer:
      greedy_q = max([self._q_values[info_state][a] for a in legal_actions])
      greedy_actions = [
          a for a in legal_actions if self._q_values[info_state][a] == greedy_q
      ]
      probs[legal_actions] = epsilon / len(legal_actions)
      probs[greedy_actions] += (1 - epsilon) / len(greedy_actions)
      action = np.random.choice(range(self._num_actions), p=probs)
    else:"""
    if False: # not is_evaluation:
      probs = np.exp(np.array([self._q_values[info_state][a] for a in legal_actions]) * self.boltzmann)
      probs = probs / np.sum(probs)
      if any(np.isnan(probs)):
        greedy_q = max([self._q_values[info_state][a] for a in legal_actions])
        greedy_actions = [
          a for a in legal_actions if self._q_values[info_state][a] == greedy_q
        ]
        probs[legal_actions] = epsilon / len(legal_actions)
        probs[greedy_actions] += (1 - epsilon) / len(greedy_actions)
        action = np.random.choice(range(self._num_actions), p=probs)
      # print("Probability: ", probs)
      else:
        action = np.random.choice(range(self._num_actions), p=probs)
    else:
      # Epsilon will be 0 when passed in
      greedy_q = max([self._q_values[info_state][a] for a in legal_actions])
      greedy_actions = [
        a for a in legal_actions if self._q_values[info_state][a] == greedy_q
      ]
      probs[legal_actions] = epsilon / len(legal_actions)
      probs[greedy_actions] += (1 - epsilon) / len(greedy_actions)
      action = np.random.choice(range(self._num_actions), p=probs)
    return action, probs

  def _get_action_probs(self, info_state, legal_actions, epsilon, is_evaluation):
    """Returns a selected action and the probabilities of legal actions.

    To be overwritten by subclasses that implement other action selection
    methods.

    Args:
      info_state: hashable representation of the information state.
      legal_actions: list of actions at `info_state`.
      epsilon: float: current value of the epsilon schedule or 0 in case
        evaluation. QLearner uses it as the exploration parameter in
        epsilon-greedy, but subclasses are free to interpret in different ways
        (e.g. as temperature in softmax).
    """
    return self._epsilon_greedy(info_state, legal_actions, epsilon, is_evaluation)

  def step(self, time_step, is_evaluation=False):
    """Returns the action to be taken and updates the Q-values if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    if self._centralized:
      info_state = str(time_step.observations["info_state"])
    else:
      info_state = str(time_step.observations["info_state"][self._player_id])
    legal_actions = time_step.observations["legal_actions"][self._player_id]

    # Prevent undefined errors if this agent never plays until terminal step
    action, probs = None, None

    if self._prev_info_state and self._prev_action and not is_evaluation:
      new_transition = Transition(
                        info_state=self._prev_info_state,
                        action=self._prev_action,
                        reward=time_step.rewards[self._player_id],
                        next_info_state=info_state,
                        is_final_step=float(time_step.last()),
                        legal_actions=legal_actions)
      self.buffer.append(new_transition)

    # Act step: don't act at terminal states.
    if not time_step.last():
      epsilon = 0.0 if is_evaluation else self._epsilon
      action, probs = self._get_action_probs(info_state, legal_actions, epsilon, is_evaluation)
    # TODO: learn if it is the last timestep
    # else:
      # self.episode_counter += 1
      # if not self.no_buffer:
      #   self.learn(is_evaluation)

    # Learn step: don't learn during evaluation or at first agent steps.
    if self._prev_info_state and not is_evaluation:
      # if self.no_buffer:
      target = time_step.rewards[self._player_id]
      if not time_step.last():  # Q values are zero for terminal.
        target += self._discount_factor * max(
            [self._q_values[info_state][a] for a in legal_actions])

      prev_q_value = self._q_values[self._prev_info_state][self._prev_action]
      self._last_loss_value = target - prev_q_value
      self._q_values[self._prev_info_state][self._prev_action] += (
          self._step_size * self._last_loss_value)

      self._loss_values_over_steps.append(abs(self._last_loss_value))
      self._total_steps += 1
      if len(self._loss_values_over_steps) > 50000:
        self._loss_values_over_steps = self._loss_values_over_steps[1:]
      
      if self._total_steps % 50000 == 0:
        print("Average tabular q loss past 10000 steps after {} training steps player {}: {}".format(self._total_steps, self._player_id, sum(self._loss_values_over_steps) / len(self._loss_values_over_steps)))

      # else:
        # self.learn(is_evaluation)

      # Decay epsilon, if necessary.
      self._epsilon = self._epsilon_schedule.step()

      if time_step.last():  # prepare for the next episode.
        self._prev_info_state = None
        self._prev_action = None
        return

    # Don't mess up with the state during evaluation.
    if not is_evaluation:
      self._prev_info_state = info_state
      self._prev_action = action
    return rl_agent.StepOutput(action=action, probs=probs)

  """
  def learn(self, is_evaluation):
    # Learn step: don't learn during evaluation or at first agent steps.
    loss = 0
    loss_values_for_pairs = collections.defaultdict(valuedict)
    update_counts_for_pairs = collections.defaultdict(valuedict)

    # if len(self.buffer) >= self.batch_size and self.episode_counter % self.train_every_episode == 0:
    # if len(self.buffer) >= self.num_transitions_bake:
    training_indices = list(np.random.choice(len(self.buffer), min(len(self.buffer), self.batch_size)))

    # for transition in reversed(self.buffer):\
    for i in training_indices:
      transition = self.buffer[i]
      o = transition.info_state
      a = transition.action
      r = transition.reward
      next_o = transition.next_info_state
      done = transition.is_final_step
      legal_actions = transition.legal_actions

      if self._prev_info_state and not is_evaluation:
      # if not is_evaluation:
        target = r
        if not done:  # Q values are zero for terminal.
          target += self._discount_factor * max(
            [self._q_values[next_o][a] for a in legal_actions])

        prev_q_value = self._q_values[o][a]
        self._last_loss_value = target - prev_q_value
        loss += abs(self._last_loss_value)

        loss_values_for_pairs[o][a] += self._last_loss_value
        update_counts_for_pairs[o][a] += 1
        self._q_values[o][a] += (self._step_size * self._last_loss_value)

      # for o, a_to_q in loss_values_for_pairs.items():
      #   for a in a_to_q.keys():
      #     average_loss = loss_values_for_pairs[o][a] / update_counts_for_pairs[o][a]
      #     self._q_values[o][a] += (self._step_size * average_loss)
      # print("Average absolute loss this batch: {}".format(loss / len(training_indices)))
      # self.buffer = []
    return
    """

  @property
  def loss(self):
    return self._last_loss_value
