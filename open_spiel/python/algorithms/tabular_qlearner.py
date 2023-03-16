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
               step_size=0.1,
               epsilon_schedule=rl_tools.ConstantSchedule(0.2), # rl_tools.LinearSchedule(1, .1, 2000000),
               discount_factor=.99,
               centralized=False):
    """Initialize the Q-Learning agent."""
    self._player_id = player_id
    self._num_actions = num_actions
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

  def _epsilon_greedy(self, info_state, legal_actions, epsilon):
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

    # Epsilon will be 0 when passed in if is_evaluation
    greedy_q = max([self._q_values[info_state][a] for a in legal_actions])
    greedy_actions = [
      a for a in legal_actions if self._q_values[info_state][a] == greedy_q
    ]
    probs[legal_actions] = epsilon / len(legal_actions)
    probs[greedy_actions] += (1 - epsilon) / len(greedy_actions)
    action = np.random.choice(range(self._num_actions), p=probs)

    return action, probs

  def _get_action_probs(self, info_state, legal_actions, epsilon):
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
    return self._epsilon_greedy(info_state, legal_actions, epsilon)

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
      self.curr_trajectory.append(new_transition)
      self.curr_return += time_step.rewards[self._player_id]

    # Act step: don't act at terminal states.
    if not time_step.last():
      epsilon = 0.0 if is_evaluation else self._epsilon
      action, probs = self._get_action_probs(info_state, legal_actions, epsilon)

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
        print("Average tabular q loss past 50000 steps after {} training steps player {}: {}".format(self._total_steps, self._player_id, sum(self._loss_values_over_steps) / len(self._loss_values_over_steps)))

      # Decay epsilon, if necessary.
      self._epsilon = self._epsilon_schedule.step()

      if time_step.last():  # prepare for the next episode.
        self._prev_info_state = None
        self._prev_action = None
        if len(self.top_return_to_trajectories) < 100 or self.curr_return > min(self.top_return_to_trajectories.keys()):
          if len(self.top_return_to_trajectories) >= 100:
            del self.top_return_to_trajectories[min(self.top_return_to_trajectories.keys())]
          self.top_return_to_trajectories[self.curr_return] = self.curr_trajectory
        self.curr_trajectory = []
        self.curr_return = 0
        return

    # Don't mess up with the state during evaluation.
    if not is_evaluation:
      self._prev_info_state = info_state
      self._prev_action = action
    return rl_agent.StepOutput(action=action, probs=probs)

  @property
  def loss(self):
    return self._last_loss_value
