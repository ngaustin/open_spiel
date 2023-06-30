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

import collections
import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import time
import random
from itertools import product

from open_spiel.python import rl_agent
from open_spiel.python import simple_nets
from open_spiel.python.utils.replay_buffer import ReplayBuffer
from open_spiel.python.algorithms.psro_v2 import utils
from open_spiel.python.algorithms.imitation_fine_tune import ImitationFineTune

# Temporarily disable TF2 behavior until code is updated.
tf.disable_v2_behavior()

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward ret")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9


class Imitation(rl_agent.AbstractAgent):
    """

    Behavior Cloning implemented with FF network

    """

    def __init__(self,
                 player_id,
                 consensus_kwargs,
                 num_actions,
                 state_representation_size,
                 num_players,
                 turn_based,
                 prev_policy, 
                 policy_constraint):

        # This call to locals() is used to store every argument used to initialize
        # the class instance, so it can be copied with no hyperparameter change.
        self._kwargs = locals()
        self.session = consensus_kwargs["session"]
        self.device = consensus_kwargs["device"]
        self._is_turn_based = turn_based

        self.player_id = player_id
        self.symmetric = consensus_kwargs["symmetric"]
        self._num_actions = num_actions
        self.state_representation_size = state_representation_size
        self.joint_action = consensus_kwargs["joint_action"]
        self.num_players = num_players

        # Controls temperature that can shift action selection towards greedy or towards random
        self.boltzmann = consensus_kwargs["boltzmann"]
        self.mode = consensus_kwargs["imitation_mode"]

        self.epochs = consensus_kwargs["training_epochs"]
        self.minimum_entropy = consensus_kwargs["minimum_entropy"]
        self.lr = consensus_kwargs["deep_network_lr"]
        self.batch = consensus_kwargs["batch_size"]
        self.hidden_layer_size = consensus_kwargs["hidden_layer_size"]
        self.n_hidden_layers = consensus_kwargs["n_hidden_layers"]
        self.discount_factor = consensus_kwargs["discount"]


        self.layer_sizes = [self.hidden_layer_size] * self.n_hidden_layers

        # For joint space stuff
        # For training in joint space

        '''
            Assumptions: 2-player game in which each player has access to the same action set.
        '''
        # Given decentralized, local action --> get joint actions that have that individual action
        # e.g. player 1 action 1 is in joint actions 1,8,16 etc.
        self.player_marginal_indices = [[[] for _ in range(num_actions)] for _ in range(num_players)]  # action index -> list of indices for marginalization
        # joint action index --> Individual action indices
        # e.g. Joint action 14, is composed of player 0 action 4 and player 1 action 2
        self.joint_index_to_joint_actions = {}  # maps joint_action_index -> all players' marginalized actions
        max_num_actions = self._num_actions ** num_players

        all_actions = list(product(list(range(num_actions)), repeat=num_players))
        print("THERE ARE A TOTAL OF {} ACTIONS IN JOINT SPACE".format(len(all_actions)))
        print("Actions here: {}".format(all_actions))

        # Initializes player_marginal indices and joint_index_to_joint_actions
        for joint_action in all_actions:
            joint_action_index = self._calculate_joint_action_index(np.array(joint_action).reshape(1, -1))
            for p in range(num_players):
                marginalized_action = joint_action[p]  # in symmetric case, self.player_id will ALWAYS be 0
                self.player_marginal_indices[p][marginalized_action].append(int(joint_action_index[0][0]))
            self.joint_index_to_joint_actions[joint_action_index[0][0]] = joint_action

        self.player_marginal_indices = np.array(self.player_marginal_indices, dtype=np.dtype(int))

        # Just holds labeled data - think of the same as an array
        self._replay_buffer = ReplayBuffer(np.inf)

        # Initialize the FF network
        num_outputs = self._num_actions ** num_players if self.joint_action else self._num_actions
        # Idea of FF Net: Map an optimal joint/independent action to every state
        # input size: num states
        # # hidden layers: layer_sizes
        # output_size: If joint actions, then num output is the total # action combinations
        self.net = simple_nets.MLP(self.state_representation_size,
                                   self.layer_sizes, num_outputs)

        self._fine_tune_mode = False

        self._variables = self.net.variables[:]

        # (observation, action) for training
        # testing: not using OOD d.p.
        if self.mode == "prob_action":
            # Allows you to pass in data later on
            self._info_state_ph = tf.placeholder(
                shape=[None, self.state_representation_size],
                dtype=tf.float32,
                name="info_state_ph")
            self._action_ph = tf.placeholder(
                shape=[None], dtype=tf.int32, name="action_ph")

            # self._return_ph = tf.placeholder(
            #     shape=[None], dtype=tf.float32, name="return_ph"
            # )

            self.log_probs = self.net(self._info_state_ph)
            loss_class = tf.losses.softmax_cross_entropy

            # Convert the actions to one hot vectors
            self.one_hot_vectors = tf.one_hot(self._action_ph, depth=num_outputs)

            # Plug into cross entropy class
            self._loss = tf.reduce_mean(loss_class(self.one_hot_vectors, self.log_probs))# weights=tf.math.exp(self._return_ph)))

        # Ignore
        elif self.mode == "prob_reward":
            raise NotImplementedError
        else:
            raise NotImplementedError

        # FINE-TUNING

        # Initialize Adam Optimizer
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self._learn_step = self._optimizer.minimize(self._loss)


        self._fine_tune_module = ImitationFineTune(self.net,
                 player_id,
                 consensus_kwargs,
                 num_actions,
                 state_representation_size,
                 num_players,
                 self._is_turn_based, 
                 prev_policy,
                 policy_constraint)


        self._initialize()

        self.states_seen_in_evaluation = []

    def clear_state_tracking(self):
        self.states_seen_in_evaluation = []
        return

    def _calculate_joint_action_index(self, joint_action):
        # Joint_action batch. Return the action index for each
        # print(joint_action)
        indices = np.zeros((joint_action.shape[0], 1), dtype=np.int32)
        for i in range(joint_action.shape[1]):
            a = joint_action[:, i:i+1]
            diff = a * (self._num_actions ** i)
            indices = indices + diff
        return indices

    def set_to_fine_tuning_mode(self, train_best_response):
        self._replay_buffer.reset()
        self._fine_tune_module.set_to_fine_tuning_mode(train_best_response)
        self._fine_tune_mode = True

    def _initialize(self):
        initialization_weights = tf.group(
            *[var.initializer for var in self._variables])
        initialization_opt = tf.group(
            *[var.initializer for var in self._optimizer.variables()])

        self.session.run(
            tf.group(*[
                initialization_weights,
                initialization_opt,
            ]))
    # FINE-TUNING END

    def step(self, time_step, is_evaluation=False, add_transition_record=True):
        """Applying trained FF network to test data

        Args:
          time_step: an instance of rl_environment.TimeStep.
          is_evaluation: bool, whether this is a training or evaluation call.
          add_transition_record: Whether to add to the replay buffer on this step.

        Returns:
          A `rl_agent.StepOutput` containing the action probs and chosen action.
        """

        # This is a weird issue with the current code framework

        if self._fine_tune_mode:
            # Edge case: set the player_id for the fine tune mode to match this module
            self._fine_tune_module.player_id = self.player_id
            fine_tune_step = self._fine_tune_module.step(time_step, is_evaluation, add_transition_record)
            return fine_tune_step
        if self.symmetric:
            # If symmetric, then having a NOT simultaneous move implies that it is updating the empirical game. Time_step.current_player is correctly set corresponding to the player
            # However, if it is a simultaneous move, then we are working with BR. self.player_id is set manually from rl_oracle.py's sample_episode to make sure we get the right observation
            player = (time_step.current_player() if not time_step.is_simultaneous_move() else self.player_id)
        else:
            # If it's not symmetric, then each agent is given one policy corresponding to player_id
            player = self.player_id

        # Act step: don't act at terminal info states or if its not our turn.
        if (not time_step.last()) and (
                time_step.is_simultaneous_move() or
                self.player_id == time_step.current_player() or self.symmetric):


            legal_actions = time_step.observations["legal_actions"][player]

            if self.joint_action:
                with tf.device(self.device):
                    info_state = np.reshape(time_step.observations["global_state"][0], [1, -1])

                logits = self.session.run(self.log_probs, feed_dict={self._info_state_ph: info_state})[0]

                all_legal_actions = time_step.observations["legal_actions"]
                legal_actions_joint = []
                for i in range(self._num_actions ** self.num_players):
                    joint_action = self.joint_index_to_joint_actions[i]
                    individual_legal_actions = [joint_action[j] in all_legal_actions[j] for j in range(self.num_players)]
                    if all(individual_legal_actions):
                        legal_actions_joint.append(i)

                legal_logits = logits[legal_actions_joint]
                action_probs = np.exp(self.boltzmann * legal_logits) / np.sum(np.exp(self.boltzmann * legal_logits))

                joint_action = utils.random_choice(legal_actions_joint, action_probs)
                action = self.joint_index_to_joint_actions[joint_action][player]

                probs = np.zeros(self._num_actions)
                probs[action] = 1.0
            else:
                info_state = time_step.observations["info_state"][player]
                with tf.device(self.device):
                    info_state = np.reshape(info_state, [1, -1])

                # Run session to get logits. Then, softmax over them
                logits = self.session.run(self.log_probs, feed_dict={self._info_state_ph: info_state})[0]

                legal_logits = logits[legal_actions]
                action_probs = np.exp(self.boltzmann * legal_logits) / np.sum(np.exp(self.boltzmann * legal_logits))

                action = utils.random_choice(legal_actions, action_probs)

                probs = np.zeros(self._num_actions)
                probs[action] = 1.0

        else:
            action = None
            probs = []

        return rl_agent.StepOutput(action=action, probs=probs)

    def add_trajectory(self, trajectory, action_trajectory, override_symmetric=False):
        """Trajectory is a list of timesteps, Action_trajectory is a list of lists representing joint actions. If it is a single player playing an action,
            it will be a list of length 1 lists. """
        """ Adds the trajectory consisting only of transitions relevant to the current player (only self.player_id if turn-based or all of them if simultaneous)"""
        if self._fine_tune_mode:
            assert False
            # self._fine_tune_module.add_trajectory(trajectory, action_trajectory, override_symmetric)
            return

        rewards_to_go = [np.zeros(self.num_players) for _ in range(len(trajectory) - 1)]
        curr_rtg = 0.0
        for i in range(len(trajectory) - 1, 0, -1):
            curr_reward = np.array(trajectory[i].rewards)  # rewards for both players
            curr_rtg = curr_reward + self.discount_factor * curr_rtg
            rewards_to_go[i-1] = curr_rtg

        for i in range(len(trajectory) - 1):
            if not self._is_turn_based:
                # NOTE: If is_symmetric, then add_transition will add observations/actions from BOTH players already
                # NOTE: Also insert action_trajectory[i+1]. If it is the last i, then we let action be 0 because it won't be used anyway
                next_action = action_trajectory[i+1] if (i+1) < len(action_trajectory) else [0 for _ in range(self.num_players)]
                self.add_transition(trajectory[i], action_trajectory[i], trajectory[i+1], next_action, ret=rewards_to_go[i], override_symmetric=override_symmetric)
            else:
                # NOTE: Assume that anything called using add_trajectory already filters out for the relevant transitions 
                player = trajectory[i].observations["current_player"]
                next_action = action_trajectory[i+1] if (i+1) < len(action_trajectory) else [0 for _ in range(self.num_players)] 
                self.add_transition(trajectory[i], action_trajectory[i], trajectory[i+1], next_action, ret=rewards_to_go[i], gae=gae[i], override_player=[player])
                """
                if player != self.player_id and not self.symmetric:
                    continue

                # Individual player's move
                action = [0 for _ in range(self.num_players)]
                next_action = [0 for _ in range(self.num_players)]

                action[player] = action_trajectory[i][0]

                # Simply indexing i+1 is incorrect. No guarantees it is this player's move or the final
                next_player_timestep_index = None
                for j in range(i+1, len(trajectory)):
                    if trajectory[j].observations["current_player"] == player or trajectory[j].last():
                        next_player_timestep_index = j
                        break
                if next_player_timestep_index:
                    next_action[player] = action_trajectory[next_player_timestep_index][0] if next_player_timestep_index < len(action_trajectory) else 0

                    curr_policy.add_transition(trajectory[i], action, trajectory[next_player_timestep_index], next_action, ret=rewards_to_go[i], override_player=[player])"""


    def add_transition(self, prev_time_step, prev_action, time_step, action, ret, override_symmetric=False, override_player=[]):
        """Adds the new transition using `time_step` to the replay buffer.

        Adds the transition from `self._prev_timestep` to `time_step` by
        `self._prev_action`.

        Args:
          prev_time_step: prev ts, an instance of rl_environment.TimeStep.
          prev_action: list of ints, joint action taken at `prev_time_step`.
          time_step: current ts, an instance of rl_environment.TimeStep.
          action: list of ints, joint action taken at timestep
          ret: return of the trajectory associated with
          override_player: player this trajectory is associated with. If empty, then default.
        """
        # TODO: If symmetric, then add transitions from both players
        if self.joint_action:
            o = prev_time_step.observations["global_state"][0][:]
            r = sum(time_step.rewards)
            store_action = int(self._calculate_joint_action_index(np.array(prev_action).reshape(1, -1))[0][0])
            transition = Transition(
                    info_state=o,
                    action=store_action,
                    reward=r,
                    ret=ret)
            self._replay_buffer.add(transition)
        else:
            player_list = [i for i in range(self.num_players)] if self.symmetric and not override_symmetric else [self.player_id]
            if len(override_player) > 0:
                player_list = override_player

            for p in player_list:
                o = prev_time_step.observations["info_state"][p][:]
                r = sum(time_step.rewards) # WELFARE

                transition = Transition(
                    info_state=(
                        prev_time_step.observations["info_state"][p][:]),
                    action=prev_action[p],
                    reward=r,
                    ret=ret)

                self._replay_buffer.add(transition)

    def _return_normalization(self, rets, temp = 1):
        """
            Softmax normalization to compute the relative weightings of the trajectories.
            Params: 
                rets: array of tuples: (player 0 future returns, player 1 future returns)
                temp: temperature hyperparameter: Can reduce/increase the power of the weighting

        """
        rets = np.array(rets)
        player_0_sum = np.sum(np.exp(rets[:, 0] / temp))
        player_1_sum = np.sum(np.exp(rets[:, 1] / temp))
        weighted_trajectories = [[np.exp(p0_rets / temp) / player_0_sum, np.exp(p1_rets / temp) / player_1_sum] for p0_rets, p1_rets in rets]
        average_weighted = [np.mean(trajectory_weight) for trajectory_weight in weighted_trajectories]
        return average_weighted

    def _select_transitions(self, dataset, weights, size_output, rng):
        index_choices = rng.choice(np.arange(len(dataset)), p=weights, size=size_output, replace=False)
        sample_transitions = []
        for index in index_choices:
            sample_transitions.append(dataset[index])
        return sample_transitions


    def learn(self):
        """Compute the loss on sampled transitions and perform a Q-network update.

        If there are not enough elements in the buffer, no loss is computed and
        `None` is returned instead.

        Returns:
          The average loss obtained on this batch of transitions or `None`.
        """

        length = len(self._replay_buffer)
        dataset = self._replay_buffer.sample(length)  # samples without replacement so take the entire thing. Random order
        for ep in range(self.epochs):
            i, batches, loss_total, entropy_total = 0, 0, 0, 0  # entropy_total is just an estimate
            dataset = random.sample(dataset, len(dataset))
            rets = [d.ret for d in dataset]
            weights = self._return_normalization(rets,temp=1)
            while i < length:
                transitions = dataset[i: min(length, i+self.batch)]
                # transitions = random.choices(population=dataset, weights=weights, k=min(length, i+self.batch) - i)
                with tf.device(self.device):
                    info_states = [t.info_state for t in transitions]
                    actions = [t.action for t in transitions]
                    rewards = [t.reward for t in transitions]
                rets = [t.ret for t in transitions]

                if self.mode == "prob_action":
                    # Session is for the FF net to learn
                    loss, _, log_probs, one_hots = self.session.run(
                    [self._loss, self._learn_step, self.log_probs, self.one_hot_vectors],
                    feed_dict={
                        self._info_state_ph: info_states,
                        self._action_ph: actions,
                        # self._return_ph: rets,
                    })

                    # Just to track entropy
                    # loss = self.session.run(
                    #     [self._loss],
                    # feed_dict={
                    #     self._info_state_ph: info_states,
                    #     self._action_ph: actions,
                    #     self._return_ph: [0 for _ in rets],
                    # })[0]
                    # print("ACTIONS: {}".format(actions))
                    # print("LOG PROBS: {}, ONE HOTS: {}".format(log_probs, one_hots))
                    # print(yes)
                elif self.mode == "prob_reward":
                    loss, _ = self.session.run(
                        [self._loss, self._learn_step],
                        feed_dict={
                            self._info_state_ph: info_states,
                            self._action_ph: actions,
                            #self._reward_ph: rewards,
                        })

                else:
                    raise NotImplemented
                loss_total += loss
                i += self.batch
                batches +=1
            if ep % 10 == 0:
                print("Average loss for epoch {}: {} ".format(ep, loss_total / float(batches)))

            if loss_total / float(batches) < self.minimum_entropy:
                print("Exiting training after {} epochs with loss of {}".format(ep, loss_total / float(batches)))
                break
        return loss

    def get_weights(self):
        #   : Implement this
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
