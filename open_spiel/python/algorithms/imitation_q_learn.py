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
    "info_state action reward next_info_state is_final_step")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9


class Imitation(rl_agent.AbstractAgent):
    """DQN Agent implementation in TensorFlow.

    See open_spiel/python/examples/breakthrough_dqn.py for an usage example.
    """

    def __init__(self,
                 player_id,
                 consensus_kwargs,
                 num_actions):
        """Initialize the DQN agent."""

        # This call to locals() is used to store every argument used to initialize
        # the class instance, so it can be copied with no hyperparameter change.
        self._kwargs = locals()

        self.player_id = player_id
        self._num_actions = num_actions
        self._epochs = consensus_kwargs["training_epochs"]

        self.observations = []  # list of observations
        self.q_values = []  # for each obs, we have a dictionary that maps each action to a DICTIONARY of q values.
                            # Maps a frozenset of other player actions to q value
        self.trajectory = []  # list of tuples. Each tuple is (prev observation, prev action, reward, observation, done)

        self.action_conditioned = True
        self.joint = consensus_kwargs["joint"]
        self.lr = 1e-2
        self.discount_factor = .99

        self.running_steps = 0
        self.running_not_seen_steps = 0

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

            if self.joint:
                info_state = np.hstack(time_step.observations["info_state"])
            else:
                info_state = time_step.observations["info_state"][self.player_id]

            legal_actions = time_step.observations["legal_actions"][self.player_id]

            obs_has_been_seen = [np.all(o == info_state) for o in self.observations]
            if True in obs_has_been_seen:
                index_observation = [np.all(o == info_state) for o in self.observations].index(True)
                q_values = self.q_values[index_observation]
                # Keep in mind not all the legal_actions might be have a key in q_values

                # This is for optimistic Q-learning
                if self.action_conditioned:
                    action_to_max_over_other_actions = {a: max(q_set.values()) for a, q_set in q_values.items()}

                    # Making it stochastic
                    """
                    actions = []
                    probs_subset = []
                    for a, q in action_to_max_over_other_actions.items():
                        actions.append(a)
                        probs_subset.append(10 * np.exp(q))
                    
                    probs_subset = np.array(probs_subset) / sum(probs_subset)
                    action_index = np.random.choice(len(probs_subset), p=probs_subset)
                    action = actions[action_index]

                    probs = []
                    for a in legal_actions:
                        if a in actions:
                            probs.append(probs_subset[actions.index(a)])
                        else:
                            probs.append(0)
                    """
                    action = max(action_to_max_over_other_actions, key=action_to_max_over_other_actions.get)
                else:
                    action = max(q_values, key=q_values.get)
                probs = []
                for a in legal_actions:
                    probs.append(1 if a == action else 0)

            else:
                # TODO: Get rid of this because for testing
                self.running_not_seen_steps += 1

                action = np.random.choice(legal_actions)
                probs = [1.0/len(legal_actions) for _ in range(len(legal_actions))]
        else:
            action = None
            probs = []

        return rl_agent.StepOutput(action=action, probs=probs)

    def add_transition(self, prev_time_step, prev_action, time_step, ret):
        """Adds the new transition using `time_step` to the replay buffer.

        Adds the transition from `self._prev_timestep` to `time_step` by
        `self._prev_action`.

        Args:
          prev_time_step: prev ts, an instance of rl_environment.TimeStep.
          prev_action: int, action taken at `prev_time_step`.
          time_step: current ts, an instance of rl_environment.TimeStep.
        """
        if self.joint:
            o = np.hstack(prev_time_step.observations["info_state"])
            next_o = np.hstack(time_step.observations["info_state"])
        else:
            o = prev_time_step.observations["info_state"][self.player_id][:]
            next_o = time_step.observations["info_state"][self.player_id][:]

        new_transition = Transition(info_state=o,
            action=prev_action,
            reward=np.sum(time_step.rewards),  # make this optimize sum of rewards?
            # next_info_state=time_step.observations["info_state"][self.player_id][:],
            next_info_state=next_o,
            is_final_step=float(time_step.last()))

        has_seen_obs = any([np.all(obs == o)for obs in self.observations])
        num_players = len(prev_action)
        if self.action_conditioned:
            other_action_set = frozenset([prev_action[i] for i in range(num_players) if i != self.player_id])
            if not has_seen_obs:
                self.observations.append(o.copy())  # Insert a new observation
                self.q_values.append({prev_action[self.player_id]: {other_action_set: 0}})  # Insert a new dictionary representing the 0 q value of (o, a) pair
            else:
                obs_index = [np.all(obs == o)for obs in self.observations].index(True)
                q_values = self.q_values[obs_index]  # This is mapping action -> frozen set -> q values
                if prev_action[self.player_id] not in q_values.keys():
                    q_values[prev_action[self.player_id]] = {other_action_set: 0}
                elif other_action_set not in q_values[prev_action[self.player_id]].keys():
                    curr_sets_to_q = q_values[prev_action[self.player_id]]
                    curr_sets_to_q[other_action_set] = 0
        else:
            if not has_seen_obs:
                self.observations.append(o.copy())
                self.q_values.append({prev_action[self.player_id]: 0})
            else:
                obs_index = [np.all(obs == o)for obs in self.observations].index(True)
                q_values = self.q_values[obs_index]
                if prev_action[self.player_id] not in q_values.keys():
                    q_values[prev_action[self.player_id]] = 0

        self.trajectory.append(new_transition)


    def learn(self):
        """Compute the loss on sampled transitions and perform a Q-network update.

        If there are not enough elements in the buffer, no loss is computed and
        `None` is returned instead.

        Returns:
          The average loss obtained on this batch of transitions or `None`.
        """
        # Iterate randomly through each element in trajectory
        print("Number of distinct observations for consensus policy: {}".format(len(self.observations)))
        self.q_values, epochs = self.do_learning_for_steps(self.q_values, self._epochs)
        print("Trained consensus for {} epochs".format(epochs))

    def do_learning_for_steps(self, q_values, epochs):
        min_loss, min_loss_epoch = np.inf, None

        for k in range(epochs):
            training_order = list(reversed(range(len(self.trajectory)))) # np.random.choice(len(self.trajectory), len(self.trajectory), replace=False)
            loss = 0
            # print("#### EPOCH {}".format(k))
            for i in training_order:
                transition = self.trajectory[i]

                o = transition.info_state
                a = transition.action
                r = transition.reward
                next_o = transition.next_info_state
                done = transition.is_final_step

                player_a = a[self.player_id]
                num_players = len(a)
                other_action_set = frozenset([a[i] for i in range(num_players) if i != self.player_id])

                # We assume that the observations and action pairs are already in the observation and q_values lists
                index_obs = [np.all(obs == o) for obs in self.observations].index(True)

                if self.action_conditioned:
                    q_value = q_values[index_obs][player_a][other_action_set]

                    if any([np.all(obs == next_o) for obs in self.observations]):
                        index_next_obs = [np.all(obs == next_o) for obs in self.observations].index(True)
                        all_q_values_joint_action = []
                        for set_to_q_values in q_values[index_next_obs].values():
                            all_q_values_joint_action.append(max(set_to_q_values.values()))

                        next_q_value_max = max(all_q_values_joint_action) # max(self.q_values[index_next_obs].values())
                    else:
                        next_q_value_max = 0
                    q_target = r + self.discount_factor * (next_q_value_max * (1 - int(done)))
                    # print("Training iteration {} q value {} q target {} reward {} next_q_value_max {}".format(i, q_value, q_target, r, next_q_value_max))
                    loss += abs(q_target - q_value)
                    new_q_value = (1 - self.lr) * (q_value) + (self.lr) * (q_target)
                    q_values[index_obs][player_a][other_action_set] = new_q_value
                else:
                    q_value = q_values[index_obs][player_a]

                    next_q_value_max = max(q_values, key=q_values.get)  # TODO: This might be incorrect
                    q_target = r + self.discount_factor * (next_q_value_max * (1 - int(done)))
                    loss += abs(q_target - q_value)
                    new_q_value = (1 - self.lr) * q_value + self.lr * q_target
                    q_values[index_obs][player_a] = new_q_value

            epoch_loss = loss / len(training_order)
            print("Training Q learn offline...epoch {} loss: ".format(k), epoch_loss)
            # print("Q values for first observation: ", q_values[0])
            # TODO: Change this eventually
            # if epoch_loss < .05:
            #     return q_values, k
            # if epoch_loss < min_loss:
            #     min_loss = epoch_loss
            #     min_loss_epoch = k+1
        """
        for i in training_order:
            transition = self.trajectory[i]
            o = transition.info_state
            index_obs = [np.all(obs == o) for obs in self.observations].index(True)
            q_value = q_values[index_obs]
            print("iteration {}".format(i), q_value)
        """
        return q_values, epochs

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
