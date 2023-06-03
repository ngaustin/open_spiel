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

# Temporarily disable TF2 behavior until code is updated.
tf.disable_v2_behavior()

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state is_final_step legal_actions_mask")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9


class JointWrapper():
  """DQN Agent implementation in TensorFlow.

  See open_spiel/python/examples/breakthrough_dqn.py for an usage example.
  """

  def __init__(self,
               agents, 
               symmetric,
               discount, 
               turn_based):
    """Initialize the DQN agent."""

    # TODO: Take in two lists: agents to train and agents to respond to 

    # Also take in the profile from the previous iteration (this should be under training parameters)

    # Create a list of agents that are CURRENTLY being trained in the upcoming episode
    
    # Also take in a variable representing psi (probability train in joint space vs BR against a different agent)

    self.agents = agents
    self._is_turn_based = turn_based
    self.num_players = len(agents)
    self.symmetric = symmetric

    self.discount = discount 
    self._prev_timestep = None 
    self._prev_action = None
    self._curr_trajectory = []
    self._action_trajectory = []


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
    action = []
    if not self._is_turn_based:
      for i, agent in enumerate(self.agents):
        agent.update_player_id(i)
        action.append(agent.step(time_step, is_evaluation, add_transition_record=False).action)   # Don't add any transitions because we manually add them from the wrapper here
    else:
      player = time_step.observations["current_player"]
      agent = self.agents[player]
      agent.update_player_id(player)
      action.append(agent.step(time_step, is_evaluation, add_transition_record=False).action)

    # Don't mess up with the state during evaluation.
    if not is_evaluation:
      # If game is symmetric, the player_id will always be 0 and prev_action and prev_timestep will be correctly assigned (because there will only be one oracle being created every iteration)
      # If the game is symmetric and we are in evaluation, prev_action and prev_obs will not be used anyway

      self._curr_trajectory.append(time_step)  # we need to also get the last timestep
      if action[0] != None:
        self._action_trajectory.append(action)  # we don't want the last action because it's always None
      """
      if self._prev_timestep and add_transition_record:
        # We may omit record adding here if it's done elsewhere.
        if self.symmetric: # only add to the first agent because they're all repeated 
            store_action = [0, 0] if time_step.last() else action  # dummy variable. Won't be used in training if last step. This is used to prevent None type errors
            # TODO: Bug. WE NEED to insert rewards to go
            self.agents[0]._policy.add_transition(self._prev_timestep, self._prev_action, time_step, store_action, np.zeros(self.num_players)) # we let ret be 0 because it won't be used in finetuning anyway
        else:
            raise NotImplementedError
      """

      if time_step.last():  # prepare for the next episode.
        self._prev_timestep = None
        self._prev_action = None

        if self.symmetric:
          self.agents[0]._policy.add_trajectory(self._curr_trajectory, self._action_trajectory)  # Takes list of timesteps and list of list of actions
        else:
          for agent in self.agents:
            agent._policy.add_trajectory(self._curr_trajectory, self._action_trajectory)

        
        self._curr_trajectory = []
        self._action_trajectory = []

        return
      else:
        self._prev_timestep = time_step
        self._prev_action = action

    return action
