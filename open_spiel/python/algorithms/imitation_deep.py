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
                 num_players):

        #Params associated with BC algorithm

        # This call to locals() is used to store every argument used to initialize
        # the class instance, so it can be copied with no hyperparameter change.
        self._kwargs = locals()
        self.session = consensus_kwargs["session"]
        self.device = consensus_kwargs["device"]

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

        self._prev_timestep = None
        self._prev_action = [None for _ in range(self.num_players)]
        self._curr_timestep = None 
        self._curr_action = [None for _ in range(self.num_players)]
        self._seen_last_time_step = 0

        self._returns = np.zeros(self.num_players)
        self._fine_tune_buffer = []
        self._fine_tune_curr_trajectory = []

        # Initialize the FF network 
        num_outputs = self._num_actions ** num_players if self.joint_action else self._num_actions 
        # Idea of FF Net: Map an optimal joint/independent action to every state
        # input size: num states
        # # hidden layers: layer_sizes 
        # output_size: If joint actions, then num output is the total # action combinations
        self.net = simple_nets.MLP(self.state_representation_size,
                                   self.layer_sizes, num_outputs)

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

        # Initialize fine tuning optimizer 
        self._rtg_ph = tf.placeholder(
                shape=[None], dtype=tf.float32, name="rtg_ph")
        self.logits = self.net(self._info_state_ph)

        probs = tf.math.exp(self.boltzmann * self.logits) / tf.reduce_sum(tf.math.exp(self.boltzmann * self.logits))
        log_probs = tf.reshape(tf.math.log(tf.gather(probs, self._action_ph, axis=1, batch_dims=1)), [-1, 1])

        rtg_weights = tf.reshape(self._rtg_ph, [-1, 1])

        assert log_probs.get_shape() == rtg_weights.get_shape()
        self._fine_tune_loss = tf.reduce_mean(-self.log_probs * rtg_weights)

        self._fine_tune_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4) 
        self._fine_tune_step = self._optimizer.minimize(self._fine_tune_loss)

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


    def _initialize(self):
        initialization_weights = tf.group(
            *[var.initializer for var in self._variables])
        initialization_opt = tf.group(
            *[var.initializer for var in self._optimizer.variables()])
        initialization_ft_opt = tf.group(
            *[var.initializer for var in self._fine_tune_optimizer.variables()]
        )

        self.session.run(
            tf.group(*[
                initialization_weights,
                initialization_opt,
                initialization_ft_opt
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

            info_state = time_step.observations["info_state"][player]
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
        '''
        if not is_evaluation: 
            if not time_step.last():
                self._curr_action[player] = action
                self._curr_timestep = time_step
            else:
                self._seen_last_time_step += 1
            
            if all(self._prev_action) and all(self._curr_action) and (player == (self.num_players - 1) or self._seen_last_time_step == self.num_players):
                # TODO: Also add if all players have seen the last time step
                self._returns = self._returns + np.array(time_step.rewards)
                self.add_fine_tune_transition(self._prev_timestep, self._prev_action, time_step, self._returns)
            
            if player == (self.num_players - 1) or self._seen_last_time_step == self.num_players:
                # Also if all players have seen the last time step
                self._prev_action = self._curr_action
                self._prev_timestep = self._curr_timestep 

                self._curr_timestep = None
                self._curr_action = [None, None]

                if time_step.last():  
                    # Add the current trajectory to the buffer 
                    self._fine_tune_buffer.append(self._fine_tune_curr_trajectory)

                    # Clear the trajectory buffer 
                    self._fine_tune_curr_trajectory = []

                    # If the buffer is up to a certain size, the train_fine_tune
                    if len(self._fine_tune_buffer) > 25: 
                        self.fine_tune()
                    
                    # prepare for the next episode.
                    self._returns = np.zeros(self.num_players)
                    self._prev_timestep = None
                    self._prev_action = [None, None]
                    self._curr_timestep = None 
                    self._curr_action = [None, None]
                    self._seen_last_time_step = 0
                    return
            '''
        return rl_agent.StepOutput(action=action, probs=probs)
    
    def add_fine_tune_transition(self, prev_time_step, prev_action, time_step, ret):
        player_list = [i for i in range(self.num_players)]  # need both players regardless of symmetric to calculate equality
        transition_list = []
        for p in player_list:
            o = prev_time_step.observations["info_state"][p][:]
            r = time_step.rewards[p]

            transition = Transition(
                info_state=(
                    prev_time_step.observations["info_state"][p][:]),
                action=prev_action[p],
                reward=r, 
                ret=ret[p])
            transition_list.append(transition)
        self._fine_tune_curr_trajectory.append(transition_list)

    def add_transition(self, prev_time_step, prev_action, time_step, action, ret):
        """Adds the new transition using `time_step` to the replay buffer.

        Adds the transition from `self._prev_timestep` to `time_step` by
        `self._prev_action`.

        Args:
          prev_time_step: prev ts, an instance of rl_environment.TimeStep.
          prev_action: tuple, action taken at `prev_time_step`.
          time_step: current ts, an instance of rl_environment.TimeStep.
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
            player_list = [i for i in range(self.num_players)] if self.symmetric else [self.player_id]

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

    def fine_tune(self):
        num_trajectories = len(self._fine_tune_buffer)
        """
        all_rets = []
        for i in range(num_trajectories):
            curr_trajectory = self._fine_tune_buffer[i]
            last_step = curr_trajectory[-1]
            rets = [i] + [last_step[j].ret for j in range(self.num_players)]
            all_rets.append(tuple(rets))

        sorted_by_player0 = sorted(all_rets, key=lambda x: (x[1])) 
        # Create list sorted by player1
        sorted_by_player1 = sorted(all_rets, key=lambda x: (x[2])) 
        top_trajectory_indices = []
            # Initialize two hashsets 
        hash_sets = [set() for _ in range(2)]

        for i in range(len(sorted_by_player0) - 1, -1, -1):
            # Add each the two trajectory indices to respective hashsets 
            hash_sets[0].add(sorted_by_player0[i][0])
            hash_sets[1].add(sorted_by_player1[i][0])
            # If the intersection between the two hashsets is non-empty, add to trajectory indices
            intersection = hash_sets[0].intersection(hash_sets[1])
            if len(intersection) > 0:
                for index in intersection:
                    top_trajectory_indices.append(index)
                hash_sets[0].remove(index)
                hash_sets[1].remove(index)
        
        batch_indices = top_trajectory_indices[:int(num_trajectories//2)]
        transitions = []
        for i in batch_indices:
            trajectory = self._fine_tune_buffer[i]
            for transition_list in trajectory:  # list of transitions corresponding to each player
                transitions.extend(transition_list)"""
        # For each of the trajectories, calculate rewards to go discounted for each of the transitions. Increase INDIVIDUAL PAYOFF (bc in this case, if we see large individual payoff, it typically isn't exploitative)

        # Add to a list of rewards to go 
        # Create a list of transitions correspondingly
        rewards_to_go = []
        transitions = []
        for trajectory in self._fine_tune_buffer:
            curr_reward_to_go = np.zeros(len(trajectory[0]))  # size num players
            for i in range(len(trajectory) - 1, -1, -1):
                curr_transition_list = trajectory[i]
                curr_rewards = np.array([t.reward for t in curr_transition_list])
                curr_reward_to_go = curr_rewards + .99 * curr_reward_to_go 
                for i, t in enumerate(curr_transition_list):
                    rewards_to_go.append(curr_reward_to_go[i])
                    transitions.append(t) 
        mean_rtg = np.mean(rewards_to_go)
        std_rtg = np.std(rewards_to_go) + 1e-7
        rewards_to_go = [(r - mean_rtg)/std_rtg for r in rewards_to_go]
        
        with tf.device(self.device):
            info_states = [t.info_state for t in transitions]
            actions = [t.action for t in transitions]
            pass_rewards_to_go = [r for r in rewards_to_go]
        
        if self.mode == "prob_action":
            loss, _, logits = self.session.run(
            [self._fine_tune_loss, self._fine_tune_step, self.logits],
            feed_dict={
                self._info_state_ph: info_states,
                self._action_ph: actions,
                self._rtg_ph: pass_rewards_to_go
                # self._return_ph: rets,
            })
        self._fine_tune_buffer = []

    def _return_normalization(self, rets, temp = 1):
        """
            Softmax normalization to compute the relative weightings of the trajectories.
            Params: 
                rets: array of tuples: (player 0 future returns, player 1 future returns)
                temp: temperature hyperparameter: Can reduce/increase the power of the weighting

        """
        rets = np.array(rets)
        player_0_sum = np.sum(np.exp(rets[:, 0]))
        player_1_sum = np.sum(np.exp(rets[:, 1]))
        weighted_trajectories = [[np.exp(p0_rets / temp) / player_0_sum, np.exp(p1_rets / temp) / player_1_sum] for p0_rets, p1_rets in rets]
        average_weighted = [np.mean(trajectory_weight) for trajectory_weight in weighted_trajectories]
        # print("AVERAGE", average_weighted)
        # print("CHECK", np.sum(average_weighted))
        return average_weighted

    def learn(self):
        """Compute the loss on sampled transitions and perform a Q-network update.

        If there are not enough elements in the buffer, no loss is computed and
        `None` is returned instead.

        Returns:
          The average loss obtained on this batch of transitions or `None`.
        """

        length = len(self._replay_buffer)
        dataset = self._replay_buffer.sample(length)  # samples without replacement so take the entire thing. Random order
        indices = list(range(length))
        for ep in range(self.epochs):
            i, batches, loss_total, entropy_total = 0, 0, 0, 0  # entropy_total is just an estimate
            dataset = random.sample(dataset, len(dataset))
            rets = [d.ret for d in dataset]
            weights = self._return_normalization(rets,temp=1)
            while i < length:
                #transitions = dataset[i: min(length, i+self.batch)] 
                transitions = random.choices(population=dataset, weights=weights, k=min(length, i+self.batch) - i)
                with tf.device(self.device):
                    info_states = [t.info_state for t in transitions]
                    actions = [t.action for t in transitions]
                    rewards = [t.reward for t in transitions]
                rets = [t.ret for t in transitions]
                #print("RETS", rets)
                # rets = [self._return_normalization(r) for r in rets]

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
            if ep % 100 == 0:
                print("Average loss for epoch {}: {} ".format(ep, loss_total / float(batches)))

            if loss_total / float(batches) < self.minimum_entropy:
                print("Exiting training after {} epochs with loss of {}".format(ep, loss_total / float(batches)))
                break 
        return loss 

    def get_weights(self):
        #   : Implement this
        print(self._returns)
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
