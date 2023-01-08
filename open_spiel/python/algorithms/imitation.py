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

from open_spiel.python import rl_agent
from open_spiel.python import simple_nets
from open_spiel.python.utils.replay_buffer import ReplayBuffer

# Temporarily disable TF2 behavior until code is updated.
tf.disable_v2_behavior()

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state is_final_step legal_actions_mask")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9


class Imitation(rl_agent.AbstractAgent):
    """DQN Agent implementation in TensorFlow.

    See open_spiel/python/examples/breakthrough_dqn.py for an usage example.
    """

    def __init__(self,
                 player_id,
                 imitation_mode,
                 num_actions):
        """Initialize the DQN agent."""

        # This call to locals() is used to store every argument used to initialize
        # the class instance, so it can be copied with no hyperparameter change.
        self._kwargs = locals()

        self.player_id = player_id
        self._num_actions = num_actions

        self.observations = []  # list of observations
        self.actions = []  # list of lists. For each observation, there is a set of actions
        self.rewards = []  # list of lists of lists. For each observation and action, there is a set of rewards
        self.counts = []  # a list of lists corresponding to each action

        self.mode = imitation_mode


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
        if (not time_step.last()) and (
                time_step.is_simultaneous_move() or
                self.player_id == time_step.current_player()):
            info_state = time_step.observations["info_state"][self.player_id]
            legal_actions = time_step.observations["legal_actions"][self.player_id]

            if not any([np.all(o == info_state) for o in self.observations]):
                # If we have not seen this state, then make it uniform
                action = np.random.choice(legal_actions)
                probs = [1.0 / len(legal_actions) for _ in legal_actions]
            else:
                # Otherwise, fit to the distribution of the inputted trajectories
                if self.mode == "prob_action":
                    index = [np.all(o == info_state) for o in self.observations].index(True)
                    actions = self.actions[index]
                    counts = self.counts[index]

                    total_count = sum(counts)
                    probs_subset = [float(c) / total_count for c in counts]
                    # print("Probs subset ", probs_subset, actions)
                    action = np.random.choice(actions, p=probs_subset)
                elif self.mode == "prob_reward":
                    index = [np.all(o == info_state) for o in self.observations].index(True)
                    actions = self.actions[index]
                    rewards = self.rewards[index]

                    reward_average_each_action = np.array([np.mean(r_set) for r_set in rewards])
                    probs_subset = np.exp(reward_average_each_action) / np.sum(np.exp(reward_average_each_action))
                    action = np.random.choice(actions, p=probs_subset)
                else:
                    print('Invalid mode for imitation')
                    assert False

                # Keep in mind some actions might be a subset of the legal_actions
                probs = []
                for i, legal_a in enumerate(legal_actions):
                    if any([np.all(legal_a == a) for a in actions]):
                        index_in_subset = [np.all(legal_a == a) for a in actions].index(True)
                        probs.append(probs_subset[index_in_subset])
                    else:
                        probs.append(0)
        else:
            action = None
            probs = []

        return rl_agent.StepOutput(action=action, probs=probs)

    def add_transition(self, prev_time_step, prev_action, time_step):
        """Adds the new transition using `time_step` to the replay buffer.

        Adds the transition from `self._prev_timestep` to `time_step` by
        `self._prev_action`.

        Args:
          prev_time_step: prev ts, an instance of rl_environment.TimeStep.
          prev_action: int, action taken at `prev_time_step`.
          time_step: current ts, an instance of rl_environment.TimeStep.
        """

        o = prev_time_step.observations["info_state"][self.player_id][:]
        r = sum(time_step.rewards) # WELFARE
        prev_action = prev_action[self.player_id]
        if any([np.all(obs == o) for obs in self.observations]):
            index = [np.all(obs == o) for obs in self.observations].index(True)
            actions_so_far = self.actions[index]
            if any([np.all(prev_action == a) for a in actions_so_far]):
                # Seen both the observation and action before
                index_action = [np.all(prev_action == a) for a in actions_so_far].index(True)
                self.counts[index][index_action] += 1
                self.rewards[index][index_action].append(r)
            else:
                # Seen the observation but NOT the action
                actions_so_far.append(prev_action)
                self.counts[index].append(1)  # 1 count now
                self.rewards[index].append([r])
        else:
            # Seen neither the observation nor the action
            self.observations.append(o)
            self.actions.append([prev_action])
            self.counts.append([1])
            self.rewards.append([[r]])


    def learn(self):
        """Compute the loss on sampled transitions and perform a Q-network update.

        If there are not enough elements in the buffer, no loss is computed and
        `None` is returned instead.

        Returns:
          The average loss obtained on this batch of transitions or `None`.
        """
        return

    def get_weights(self):
        # TODO: Implement this
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
