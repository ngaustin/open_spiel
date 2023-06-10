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
from itertools import product
import random
import time

from open_spiel.python import rl_agent
from open_spiel.python import simple_nets
from open_spiel.python.utils.replay_buffer import ReplayBuffer
from open_spiel.python.algorithms.psro_v2 import utils

# Temporarily disable TF2 behavior until code is updated.
tf.disable_v2_behavior()

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state next_action is_final_step legal_actions_mask rewards_to_go gae")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9

# Since we are working in joint action space, the output of the Q network will be exponential in the number of players 
# We index it so that for a player with player index of i, their action will change every (num_actions) ** i indices in the action output 
# For example, if a game has an action space size of 2, for player index 0, their action will be different for every 1 of the outputs 
# For player index 1, their action will be different every 2 outputs 
# For player index 2, their action will be different every 4 outputs

# To marginalize a player's action, take slices of size (num_actions) ** i every (num_actions) ** (i + 1) indices. 
# Then, take the max of them. That represents the marginalized max Q-value of that actions

# To get different marginalized Q-value for each action, start at different points: 0, num_actions ** i, 2 * (num_actions ** i), 3 * (num_actions ** i) etc.

# To calculate the Q value for a particular joint action, just take indices of all of the players and calculate the number in base num_actions

class RunningStat(object):
    '''
    Keeps track of first and second moments (mean and variance)
    of a streaming time series.
     Taken from https://github.com/joschu/modular_rl
     Math in http://www.johndcook.com/blog/standard_deviation/
    '''
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape

class RewardScaler(object):
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """
    def __init__(self, shape, center=True, scale=True):
        assert shape is not None
        self.center = center
        self.scale = scale
        self.rs = RunningStat(shape)
    
    def push(self, x):
        self.rs.push(x)

    def normalize(self, x):
        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + 1e-8)
            else:
                diff = x - self.rs.mean
                diff = diff/(self.rs.std + 1e-8)
                x = diff + self.rs.mean
        return x
    
    def denormalize(self, x):
        if self.scale:
            if not self.center:
                diff = x - self.rs.mean 
                diff = diff * (self.rs.std + 1e-8)
                x = diff + self.rs.mean 
            else:
                x = x * (self.rs.std + 1e-8)
        if self.center:
            x = x + self.rs.mean
    
        return x 



class ImitationFineTune(rl_agent.AbstractAgent):
    """Conservative Q-Learning Agent implementation in TensorFlow.

    See open_spiel/python/examples/breakthrough_dqn.py for an usage example.
    """

    def __init__(self,
                 pre_trained_network,
                 player_id,
                 consensus_kwargs,
                 num_actions, 
                 state_representation_size, 
                 num_players, 
                 turn_based):
        """Initialize the DQN agent."""

        # This call to locals() is used to store every argument used to initialize
        # the class instance, so it can be copied with no hyperparameter change.
        self._kwargs = locals()
        self.session = consensus_kwargs["session"]
        self.num_players = num_players
        self.device = consensus_kwargs["device"]
        self._is_turn_based = turn_based

        self.player_id = player_id
        self.symmetric = consensus_kwargs["symmetric"]
        self._num_actions = num_actions

        self.lr = consensus_kwargs["deep_network_lr"]
        self.policy_lr = consensus_kwargs["deep_policy_network_lr"]
        self.batch = consensus_kwargs["batch_size"]
        self.hidden_layer_size = consensus_kwargs["hidden_layer_size"]
        self.n_hidden_layers = consensus_kwargs["n_hidden_layers"]
        self.rewards_joint = consensus_kwargs["rewards_joint"]
        self.joint_action = consensus_kwargs['joint_action']
        self.discount_factor = consensus_kwargs["discount"]
        self.entropy_regularization = consensus_kwargs["ppo_entropy_regularization"]

        # BELOW is for R-BVE finetuning 
        self.max_buffer_size_fine_tune = consensus_kwargs["max_buffer_size_fine_tune"]
        self.min_buffer_size_fine_tune = consensus_kwargs["min_buffer_size_fine_tune"]
        self.fine_tune_bool = consensus_kwargs["fine_tune"]
        self.consensus_kwargs = consensus_kwargs

        self.actor_loss_list = []
        self.value_loss_list = []
        self.entropy_list = []
        self.kl_list = []

        self.layer_sizes = [self.hidden_layer_size] * self.n_hidden_layers

        # Initialize replay
        self._replay_buffer = ReplayBuffer(np.inf)
        self._all_trajectories = []
        self._all_action_trajectories = []
        self._all_override_symmetrics = []
        self._curr_size_batch = 0
        self._seen_observations = set()

        # Initialize the FF network 

        num_outputs = self._num_actions ** num_players if self.joint_action else self._num_actions 

        self.trained_once = False
        
        # Create placeholders
        self._info_state_ph = tf.placeholder(
            shape=[None, state_representation_size],
            dtype=tf.float32,
            name="info_state_ph")
        self._action_ph = tf.placeholder(
            shape=[None, 1], dtype=tf.int32, name="action_ph")
        self._is_final_step_ph = tf.placeholder(
            shape=[None], dtype=tf.float32, name="is_final_step_ph")
        self._next_info_state_ph = tf.placeholder(
            shape=[None, state_representation_size],
            dtype=tf.float32,
            name="next_info_state_ph")
        self._rewards_to_go_ph = tf.placeholder(
            shape=[None], dtype=tf.float32, name="rewards_to_go_ph"
        )
        self._next_action_ph = tf.placeholder(
            shape=[None, 1], dtype=tf.int32, name="next_action_ph"
        )


        ####### R-BVE/SARSA Start ########
        self._fine_tune_mode_ph = tf.placeholder(
            shape=(),
            dtype=tf.bool,
            name="fine_tune_mode_ph")
        
        self._fine_tune_steps = 0
        self._env_steps = 0
        self._fine_tune_print_counter = 0

        ##################################################################

        ################### Begin PPO fine-tuning code ###################

        ##################################################################
        self._old_log_probs_ph = tf.placeholder(
            shape=[None, 1],
            dtype=tf.float32,
            name="old_log_probs_ph"
        )
        self._gae_ph = tf.placeholder(
            shape=[None, 1],
            dtype=tf.float32,
            name="gae_ph"
        )

        self._policy_constraint_weight_ph = tf.placeholder(
            shape=(),
            dtype=tf.float32,
            name="policy_constraint_weight_ph"
        )
        # Trajectories and actions for online fine-tuning without the joint wrapper 
        self.curr_trajectory = []
        self.action_trajectory = []

        # Create a policy network same size as Q network 
        self._policy_network = simple_nets.MLP(state_representation_size, 
                                            self.layer_sizes, num_outputs)
        self._policy_network_variables = self._policy_network.variables[:]
        self._old_policy_network = simple_nets.MLP(state_representation_size, 
                                            self.layer_sizes, num_outputs)
        self._old_policy_network_variables = self._policy_network.variables[:]
        # Create a VALUE network same size as Q network
        self._value_network = simple_nets.MLP(state_representation_size, 
                                    self.layer_sizes, 1)
        self._value_network_variables = self._value_network.variables[:]

        self.reward_scaler = RewardScaler(shape=self.num_players, center=True, scale=True)
    
        # Create a method that copies parameters from Q network to policy network 
        self._initialize_policy_network = self._create_policy_network(self._policy_network, pre_trained_network)
        self._initialize_old_policy_network = self._create_policy_network(self._old_policy_network, pre_trained_network)

        # Pass observations to policy 
        logits = self._policy_network(self._info_state_ph) # [?, num_actions]

        # Get the outputs...softmax over them 
        # exps = tf.math.exp(logits) # [?, num_actions]
        # normalizer = tf.reshape(tf.reduce_sum(exps, axis=1), [-1, 1]) # [?, 1]
        self.probs = tf.nn.softmax(logits, axis=1)  # exps / normalizer # [?, num_actions]
        # print("Normalizer: ", normalizer.get_shape(), "  probs: ", exps.get_shape(), self.probs.get_shape())

        # Then do tf.log on them to get logprobs 
        all_log_probs = tf.math.log(tf.clip_by_value(self.probs, 1e-10, 1.0)) # [?, num_actions]
        self.log_probs = tf.gather(all_log_probs, self._action_ph, axis=1, batch_dims=1) # [?, 1]

        ###################################
        #### Policy constraint section ####
        constrain_logits = self._old_policy_network(self._info_state_ph)
        constrain_probs = tf.nn.softmax(constrain_logits, axis=1)

        constrain_all_log_probs = tf.stop_gradient(tf.math.log(tf.clip_by_value(constrain_probs, 1e-10, 1.0)))
        self.constrain_kl_divergence = tf.reduce_mean(tf.reduce_sum(tf.math.multiply(tf.math.exp(constrain_all_log_probs), constrain_all_log_probs - all_log_probs), axis=1))
        ###################################

        # Pass observations to value network to get value baseline 
        self.values = self._value_network(self._info_state_ph) # [?, 1]

        # Subtract from the rewards to go to get advantage values 
        assert (tf.reshape(self._rewards_to_go_ph, [-1, 1])).get_shape() == self.values.get_shape()
        value_delta = tf.reshape(self._rewards_to_go_ph, [-1, 1]) - self.values  # [?, 1]

        advantages = tf.reshape(self._gae_ph, [-1, 1])

        # Calculate entropy 
        assert self.probs.get_shape() == all_log_probs.get_shape()
        self.entropy = tf.reduce_mean(-(tf.reduce_sum(self.probs * all_log_probs, axis=1)))

        # Calculate actor loss by negative weighting log probs by DETACHED advantage values and adding in entropy regularization with .01 weight
        assert self.log_probs.get_shape() == advantages.get_shape()
        normalized_advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-7)

        # Calculate PPO actor loss 
        eps_clip = consensus_kwargs["eps_clip"]
        assert self.log_probs.get_shape() == tf.reshape(tf.stop_gradient(self._old_log_probs_ph), [-1, 1]).get_shape()
        ratios = tf.math.exp(self.log_probs - tf.reshape(tf.stop_gradient(self._old_log_probs_ph), [-1, 1]))
        assert ratios.get_shape() == normalized_advantages.get_shape() 
        surr1 = ratios * tf.stop_gradient(normalized_advantages) 
        surr2 = tf.clip_by_value(ratios, 1-eps_clip, 1+eps_clip) * tf.stop_gradient(normalized_advantages)
        self.actor_loss = -tf.reduce_mean(tf.math.minimum(surr1, surr2)) - (self.entropy_regularization * self.entropy) + (self._policy_constraint_weight_ph * self.constrain_kl_divergence)

        # Calculcate critic loss by taking the square of advantages 
        self.critic_loss = tf.reduce_mean(tf.math.square(value_delta)) 

        # Create separate optimizers for the policy and value network 
        self._ppo_policy_optimizer = tf.train.AdamOptimizer(learning_rate=3e-5)
        self._ppo_value_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

        # Learn step
        # self._ppo_learn_step = self._ppo_optimizer.minimize(.5 * self.critic_loss + self.actor_loss)
        self._ppo_value_learn_step = self._ppo_value_optimizer.minimize(self.critic_loss)
        self._ppo_policy_learn_step = self._ppo_policy_optimizer.minimize(self.actor_loss)

        # Change step to query the policy network instead of the q network
        # Change initialize method to include the new networks and the new optimizers

        self._initialize()

        self.states_seen_in_evaluation = []

    def clear_state_tracking(self):
        self.states_seen_in_evaluation = []
        return 
    
    # TODO: Make sure this transitions is smooth
    def _create_policy_network(self, policy_network, source_network):
        self._variables = source_network.variables[:]
        self._policy_variables = policy_network.variables[:]
        assert self._variables
        assert len(self._variables) == len(self._policy_variables)
        return tf.group([
            tf.assign(target_v, v)#  self.tau * v + (1 - self.tau) * target_v) #
            for (target_v, v) in zip(self._policy_variables, self._variables)
        ])

    def set_to_fine_tuning_mode(self, is_train_best_response):
        self._replay_buffer.reset()
        self._replay_buffer = ReplayBuffer(self.max_buffer_size_fine_tune)
        self.session.run(self._initialize_policy_network)
        self.session.run(self._initialize_old_policy_network)

        self.epochs = 10  # consider using 30 epochs here as well
        self.minibatches = 50
        if is_train_best_response:
            new_checkpoint = tf.train.Checkpoint(model=self._policy_network)
            new_checkpoint.restore("ppo_best_response_pretrain_starting_point")
            print('Pre-trained best response PPO policy network checkpoint restored.')

            # self.epochs = 10  # consider using 30 epochs here as well
            # self.minibatches = 50

            self.policy_constraint_weight = 0
        else:
            # self.epochs = 5
            # self.minibatches = 1
            # self.min_buffer_size_fine_tune = 6000

            # self.epochs = 30
            # self.minibatches = 30 
            # self.min_buffer_size_fine_tune = 120000

            self.policy_constraint_weight = self.consensus_kwargs["policy_constraint"]

            # Consider making minibatches = 30, self.min_buffer_size_fine_tune = 120000, and self.epochs = 30
            # self.min_buffer_size_fine_tune *= 2

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
                self.player_id == time_step.current_player() or self.symmetric):

            # This is a weird issue with the current code framework
            if self.symmetric:
                # If symmetric, then having a NOT simultaneous move implies that it is updating the empirical game. Time_step.current_player is correctly set corresponding to the player 
                # However, if it is a simultaneous move, then we are working with BR. self.player_id is set manually from rl_oracle.py's sample_episode to make sure we get the right observation
                player = (time_step.current_player() if not time_step.is_simultaneous_move() else self.player_id)
            else:
                # If it's not symmetric, then each agent is given one policy corresponding to player_id
                player = self.player_id 
                
            info_state = time_step.observations["info_state"][player]
            
            # with tf.device(self.device):
            info_state = np.reshape(info_state, [1, -1])
                

            if self.joint_action:
                raise NotImplementedError 
                
            else:
                # probs = np.zeros(self._num_actions)
                legal_actions = time_step.observations["legal_actions"][player]

                probs = self.session.run(self.probs, feed_dict={self._info_state_ph:info_state})[0]
                legal_probs = probs[legal_actions]
                legal_probs = legal_probs / np.sum(legal_probs)
                action = utils.random_choice(legal_actions, legal_probs)

                probs[action] = 1.0
            # TODO: Remove this. THis is for experiment 
            # if is_evaluation:
            #     self.states_seen_in_evaluation.append(''.join(map(str, time_step.observations["info_state"][self.player_id])))
        else:
            action = None
            probs = []

        if not is_evaluation:
            self.curr_trajectory.append(time_step)
            if action != None:
                self.action_trajectory.append([action])

            
            if time_step.last():
                if add_transition_record:
                    self.add_trajectory(self.curr_trajectory, self.action_trajectory, override_symmetric=True)  # we want to override symmetric because we are now training individually against other targets that are not ourselves
                self.curr_trajectory = []
                self.action_trajectory = []

                if self._curr_size_batch > self.min_buffer_size_fine_tune:
                    self._fine_tune_steps += 1
                    self.insert_transitions()
                    self.fine_tune()

        return rl_agent.StepOutput(action=action, probs=probs)
    
    def fine_tune(self):
        self._env_steps += len(self._replay_buffer)

        epochs = self.epochs
        minibatches = self.minibatches

        transitions = self._replay_buffer.sample(len(self._replay_buffer))
        info_states = [t.info_state for t in transitions]
        actions = [[t.action] for t in transitions]
        """
        next_actions = [[t.next_action] for t in transitions]
        rewards = [t.reward for t in transitions]
        next_info_states = [t.next_info_state for t in transitions]
        are_final_steps = [t.is_final_step for t in transitions]
        legal_actions_mask = [t.legal_actions_mask for t in transitions]
        rewards_to_go = [t.rewards_to_go for t in transitions]
        gae = [[t.gae] for t in transitions]
        """

        old_log_probs = self.session.run(
            [self.log_probs],
            feed_dict={
                self._info_state_ph: info_states,
                self._action_ph: actions
            }
        )[0]

        indices = np.arange(len(transitions))
        size_minibatch = len(transitions) // minibatches
        # For number of minibatches 
        for _ in range(epochs):
            self._fine_tune_print_counter += 1
            # Get some random order of all of the transitions 
            np.random.shuffle(indices)


            for i in range(minibatches):
                start = i*size_minibatch 
                end = (i+1)*size_minibatch if i < minibatches - 1 else len(indices)

                subset = indices[start:end]
                info_states = [transitions[j].info_state for j in subset]
                actions = [[transitions[j].action] for j in subset]
                next_actions =[[transitions[j].next_action] for j in subset]
                rewards = [transitions[j].reward for j in subset]
                next_info_states = [transitions[j].next_info_state for j in subset]
                are_final_steps = [transitions[j].is_final_step for j in subset]
                legal_actions_mask = [transitions[j].legal_actions_mask for j in subset]
                rewards_to_go = [transitions[j].rewards_to_go for j in subset]
                gae = [[transitions[j].gae] for j in subset]
                old_log_probs_subset = [old_log_probs[j] for j in subset]

                actor_loss, _, entropy, probs, log_probs, kl = self.session.run(
                [self.actor_loss, self._ppo_policy_learn_step, self.entropy, self.probs, self.log_probs, self.constrain_kl_divergence],
                feed_dict={
                    self._info_state_ph: info_states,
                    self._action_ph: actions,
                    self._rewards_to_go_ph: rewards_to_go,
                    self._old_log_probs_ph: old_log_probs_subset,
                    self._fine_tune_mode_ph: True,
                    self._gae_ph: gae,
                    self._policy_constraint_weight_ph: self.policy_constraint_weight
                })
            
            self.actor_loss_list.append(actor_loss)
            self.entropy_list.append(entropy)
            self.kl_list.append(kl)

        # For number of minibatches 
        for _ in range(epochs):
            np.random.shuffle(indices)


            for i in range(minibatches):
                start = i*size_minibatch 
                end = (i+1)*size_minibatch if i < minibatches - 1 else len(indices)

                subset = indices[start:end]
                info_states = [transitions[j].info_state for j in subset]
                rewards_to_go = [transitions[j].rewards_to_go for j in subset]


                value_loss, _ = self.session.run(
                    [self.critic_loss, self._ppo_value_learn_step],
                    feed_dict={
                        self._info_state_ph: info_states,
                        self._rewards_to_go_ph: rewards_to_go,
                        self._fine_tune_mode_ph: True
                    })
            self.value_loss_list.append(value_loss)
        """

        # NOTE: In multi-agent settings, having one minibatch tends to work better in Independent PPO: https://arxiv.org/pdf/2103.01955.pdf . Also, having 5 or less epochs seems to work best
        for _ in range(15):
            actor_loss, _, entropy, probs, log_probs = self.session.run(
                [self.actor_loss, self._ppo_policy_learn_step, self.entropy, self.probs, self.log_probs],
                feed_dict={
                    self._info_state_ph: info_states,
                    self._action_ph: actions,
                    self._rewards_to_go_ph: rewards_to_go,
                    self._old_log_probs_ph: old_log_probs,
                    self._fine_tune_mode_ph: True,
                    self._gae_ph: gae,
                })

            if self._fine_tune_learn_steps == 1:
                print("Actor loss and entropy and log probs/probs on first fine tune steps: ", actor_loss, entropy, log_probs, probs)
                
        for _ in range(15):
            value_loss, _ = self.session.run(
                [self.critic_loss, self._ppo_value_learn_step],
                feed_dict={
                    self._info_state_ph: info_states,
                    self._rewards_to_go_ph: rewards_to_go,
                    self._fine_tune_mode_ph: True
                })
        """
        
        if (len(self.actor_loss_list) > 100 and self._fine_tune_print_counter > 100):
            self.actor_loss_list = self.actor_loss_list[-100:]
            self.value_loss_list = self.value_loss_list[-100:]
            self.entropy_list = self.entropy_list[-100:]
            self.kl_list = self.kl_list[-100:]
            print("Mean PPO Actor + Value losses, entropy, and kl last 100 updates...and num env steps: ", sum(self.actor_loss_list) / len(self.actor_loss_list), sum(self.value_loss_list) / len(self.value_loss_list), sum(self.entropy_list) / len(self.entropy_list), sum(self.kl_list) / len(self.kl_list), self._env_steps)
            # print("Reward scaling mean, std: ", self.reward_scaler.rs.mean, self.reward_scaler.rs.std)
            self._fine_tune_print_counter = 0

        # Reset everything
        self._replay_buffer.reset()
        self._all_trajectories = []
        self._all_action_trajectories = []
        self._all_override_symmetrics = []
        self._curr_size_batch = 0

        # Save the model if we want to...comment out if necessary 
        # checkpoint = tf.train.Checkpoint(model=self._policy_network)
        # checkpoint.write("ppo_best_response_pretrain_starting_point")
        return

    def _initialize(self):
        initialization_policy = tf.group(
            *[var.initializer for var in self._policy_network_variables]
        )
        initialization_value = tf.group(
            *[var.initializer for var in self._value_network_variables]
        )
        initialization_ppo_value_opt = tf.group(
            *[var.initializer for var in self._ppo_value_optimizer.variables()]
        )
        initialization_ppo_policy_opt = tf.group(
            *[var.initializer for var in self._ppo_policy_optimizer.variables()]
        )
        initialization_old_policy = tf.group(
            *[var.initializer for var in self._old_policy_network_variables]
        )
    
        # initialization_weights,
        # initialization_opt,
        self.session.run(
            tf.group(*[
                initialization_policy,
                initialization_value,
                initialization_ppo_value_opt,
                initialization_ppo_policy_opt,
                initialization_old_policy
            ]))

    def cumsum(self, x, discount):
        vals = [None for _ in range(len(x))]
        curr = 0 
        for i in range(len(x) - 1, -1, -1):
            vals[i] = curr = x[i] + discount  * curr
        return vals

    def add_trajectory(self, trajectory, action_trajectory, override_symmetric=False):
        self._curr_size_batch += len(action_trajectory)

        self._all_trajectories.append(trajectory)
        self._all_action_trajectories.append(action_trajectory)
        self._all_override_symmetrics.append(override_symmetric)

    def insert_transitions(self):
        all_gae = []
        all_rewards_to_go = []
        
        for trajectory in self._all_trajectories:
            reward_arrays = [np.array(timestep.rewards) for timestep in trajectory[1:]]  # the first timestep has None value 
            rewards_to_go = [np.zeros(self.num_players) for _ in range(len(trajectory) - 1)]

            # For each of the trajectories and action trajectories, calculate the rewards to go
            rewards_to_go = self.cumsum(reward_arrays, self.discount_factor)
            all_rewards_to_go.append(rewards_to_go)
            
            # Push the rewards to go to our running average/std calculator
            for rtg in rewards_to_go:
                self.reward_scaler.push(rtg)

        for i, trajectory in enumerate(self._all_trajectories):
            gae = [np.zeros(self.num_players) for _ in range(len(trajectory) - 1)]
            values = [np.zeros(self.num_players) for _ in range(len(trajectory))]

            for p in range(self.num_players):
                observations = [timestep.observations["info_state"][p] for timestep in trajectory]
                vals = self.session.run([self.values], feed_dict={self._info_state_ph:observations})[0]
                vals = np.array(vals)

                for j, val in enumerate(values):
                    val[p] = vals[j] 
            
            # Denormalize the values, normalize the rewards to go
            # NOTE: found performance to be much better without this normalization
            # values = [self.reward_scaler.denormalize(val) for val in values]

            # all_rewards_to_go[i] = [self.reward_scaler.normalize(rtg) for rtg in all_rewards_to_go[i]]

            reward_arrays = [np.array(timestep.rewards) for timestep in trajectory[1:]]
            deltas = [reward_arrays[j] + self.discount_factor * values[j+1] - values[j] for j in range(len(reward_arrays))]
            gae = self.cumsum(deltas, self.discount_factor * .95) 

            # Append gae and normalized rewards to go 
            all_gae.append(gae)

        for curr, (trajectory, action_trajectory) in enumerate(zip(self._all_trajectories, self._all_action_trajectories)):
            rewards_to_go = all_rewards_to_go[curr]
            gae = all_gae[curr]

            for i in range(len(trajectory) - 1):
                if not self._is_turn_based:
                    # NOTE: If is_symmetric, then add_transition will add observations/actions from BOTH players already
                    # NOTE: Also insert action_trajectory[i+1]. If it is the last i, then we let action be 0 because it won't be used anyway
                    next_action = action_trajectory[i+1] if (i+1) < len(action_trajectory) else [0 for _ in range(self.num_players)] 
                    self.add_transition(trajectory[i], action_trajectory[i], trajectory[i+1], next_action, ret=rewards_to_go[i], gae=gae[i], override_symmetric=self._all_override_symmetrics[curr]) 
                else:
                    player = trajectory[i].observations["current_player"]

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

                        curr_policy.add_transition(trajectory[i], action, trajectory[next_player_timestep_index], next_action, ret=rewards_to_go[i], override_player=[player]) 

    def add_transition(self, prev_time_step, prev_action, time_step, action, ret, gae, override_symmetric=False, override_player=[]):
        """Adds the new transition using `time_step` to the replay buffer.

        Adds the transition from `self._prev_timestep` to `time_step` by
        `self._prev_action`.

        Args:
          prev_time_step: prev ts, an instance of rl_environment.TimeStep.
          prev_action: list of int, action taken at `prev_time_step`.
          time_step: current ts, an instance of rl_environment.TimeStep.
        """
        player_list = [i for i in range(self.num_players)] if self.symmetric and not override_symmetric else [self.player_id]

        if self.joint_action: 
            raise NotImplementedError
        else:
            for p in player_list:
                o = prev_time_step.observations["info_state"][p][:]

                assert not self.rewards_joint  # because gae is not valid

                r = sum(time_step.rewards) if self.rewards_joint else time_step.rewards[p] # WELFARE
                rewards_to_go = sum(ret) if self.rewards_joint else ret[p]

                # NOTE: We want to keep all the transitions consistent...as if it was from one player's perspective.
                # So, we need to switch the actions for consistency (this implementation only works for 2-player games)
                # If we didn't switch, then the same action for the same observation can yield different results depending on which player we took that transition from 

                # Since the step() function assumes by symmetry that observations come from player0, we need to make sure that all 
                # transitions are from player0's perspective, meaning the action applied to the observed player's observation must come first

                store_action = prev_action[p]
                store_next_action = action[p]

                legal_actions = (time_step.observations["legal_actions"][p])
                legal_actions_mask = np.zeros(self._num_actions)
                legal_actions_mask[legal_actions] = 1.0

                transition = Transition(
                    info_state=(
                        prev_time_step.observations["info_state"][p][:]),
                    action=store_action,
                    reward=r, 
                    next_info_state=time_step.observations["info_state"][p][:],
                    next_action = store_next_action,
                    is_final_step=float(time_step.last()),
                    legal_actions_mask=legal_actions_mask,
                    rewards_to_go=rewards_to_go, 
                    gae=gae[p]
                )

                self._replay_buffer.add(transition)

    def get_weights(self):
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
