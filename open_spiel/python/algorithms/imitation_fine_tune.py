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
    def __init__(self, shape, gamma):
        assert shape is not None 
        self.gamma = gamma 
        self.rs = RunningStat(shape)
        self.ret = np.zeros(shape)
    
    def __call__(self, x):
        self.ret = self.ret * self.gamma + x
        self.rs.push(self.ret)
        x = x / (self.rs.std + 1e-8)
        return x 

    def reset(self):
        self.ret = np.zeros_like(self.ret)


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
                 num_players):
        """Initialize the DQN agent."""

        # This call to locals() is used to store every argument used to initialize
        # the class instance, so it can be copied with no hyperparameter change.
        self._kwargs = locals()
        self.session = consensus_kwargs["session"]
        self.num_players = num_players
        self.device = consensus_kwargs["device"]

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

        # BELOW is for R-BVE finetuning 
        self.max_buffer_size_fine_tune = consensus_kwargs["max_buffer_size_fine_tune"]
        self.min_buffer_size_fine_tune = consensus_kwargs["min_buffer_size_fine_tune"]
        self.fine_tune_bool = consensus_kwargs["fine_tune"]

        self.actor_loss_list = []
        self.value_loss_list = []
        self.entropy_list = []

        self.layer_sizes = [self.hidden_layer_size] * self.n_hidden_layers

        # Initialize replay
        self._replay_buffer = ReplayBuffer(np.inf)
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
        self._fine_tune_learn_steps = 0

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
        # Trajectories and actions for online fine-tuning without the joint wrapper 
        self.curr_trajectory = []
        self.action_trajectory = []

        # Create a policy network same size as Q network 
        self._policy_network = simple_nets.MLP(state_representation_size, 
                                            self.layer_sizes, num_outputs)
        self._policy_network_variables = self._policy_network.variables[:]
        # Create a VALUE network same size as Q network
        self._value_network = simple_nets.MLP(state_representation_size, 
                                    self.layer_sizes, 1)
        self._value_network_variables = self._value_network.variables[:]

        self.reward_scaler = RewardScaler(shape=self.num_players, gamma=self.discount_factor)
    
        # Create a method that copies parameters from Q network to policy network 
        self._initialize_policy_network = self._create_policy_network(self._policy_network, pre_trained_network)

        # Pass observations to policy 
        logits = self._policy_network(self._info_state_ph) # [?, num_actions]

        # Get the outputs...softmax over them 
        exps = tf.math.exp(logits) # [?, num_actions]
        normalizer = tf.reshape(tf.reduce_sum(exps, axis=1), [-1, 1]) # [?, 1]
        self.probs = exps / normalizer # [?, num_actions]
        print("Normalizer: ", normalizer.get_shape(), "  probs: ", exps.get_shape(), self.probs.get_shape())

        # Then do tf.log on them to get logprobs 
        all_log_probs = tf.math.log(self.probs) # [?, num_actions]
        self.log_probs = tf.gather(all_log_probs, self._action_ph, axis=1, batch_dims=1) # [?, 1]

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
        ratios = tf.math.exp(self.log_probs - tf.reshape(tf.stop_gradient(self._old_log_probs_ph), [-1, 1]))
        surr1 = ratios * tf.stop_gradient(normalized_advantages) 
        surr2 = tf.clip_by_value(ratios, 1-eps_clip, 1+eps_clip) * tf.stop_gradient(normalized_advantages)
        self.actor_loss = -tf.reduce_mean(tf.math.minimum(surr1, surr2)) # - .01 * self.entropy

        # self.actor_loss = tf.reduce_mean(self.log_probs * tf.stop_gradient(normalized_advantages)) - .01 * self.entropy
        
        # Calculcate critic loss by taking the square of advantages 
        self.critic_loss = tf.reduce_mean(tf.math.square(value_delta)) 

        # Create separate optimizers for the policy and value network 
        self._ppo_policy_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)
        self._ppo_value_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

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
    def _create_policy_network(self, policy_network, q_network):
        self._variables = q_network.variables[:]
        self._policy_variables = policy_network.variables[:]
        assert self._variables
        assert len(self._variables) == len(self._policy_variables)
        return tf.group([
            tf.assign(target_v, v)#  self.tau * v + (1 - self.tau) * target_v) #
            for (target_v, v) in zip(self._policy_variables, self._variables)
        ])

    # TODO: Possibly remove/edit this
    def set_to_fine_tuning_mode(self):
        self._replay_buffer.reset()
        self._replay_buffer = ReplayBuffer(self.max_buffer_size_fine_tune)
        self.session.run(self._initialize_policy_network)

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
                self.action_trajectory.append(action)

            
            if time_step.last():
                if add_transition_record:
                    self.add_trajectory(self.curr_trajectory, self.action_trajectory, override_symmetric=True)  # we want to override symmetric because we are now training individually against other targets that are not ourselves
                self.curr_trajectory = []
                self.action_trajectory = []

                if len(self._replay_buffer) > self.min_buffer_size_fine_tune:
                    self._fine_tune_steps += 1
                    self.fine_tune()

        return rl_agent.StepOutput(action=action, probs=probs)
    
    def fine_tune(self):
        self._fine_tune_learn_steps += 1
        transitions = self._replay_buffer.sample(len(self._replay_buffer))
        info_states = [t.info_state for t in transitions]
        actions = [[t.action] for t in transitions]
        next_actions = [[t.next_action] for t in transitions]
        rewards = [t.reward for t in transitions]
        next_info_states = [t.next_info_state for t in transitions]
        are_final_steps = [t.is_final_step for t in transitions]
        legal_actions_mask = [t.legal_actions_mask for t in transitions]
        rewards_to_go = [t.rewards_to_go for t in transitions]
        gae = [[t.gae] for t in transitions]

        old_log_probs = self.session.run(
            [self.log_probs],
            feed_dict={
                self._info_state_ph: info_states,
                self._action_ph: actions
            }
        )[0]


        # NOTE: In multi-agent settings, having one minibatch tends to work better in Independent PPO: https://arxiv.org/pdf/2103.01955.pdf . Also, having 5 or less epochs seems to work best
        for _ in range(5):
            actor_loss, _, entropy, probs = self.session.run(
                [self.actor_loss, self._ppo_policy_learn_step, self.entropy, self.probs],
                feed_dict={
                    self._info_state_ph: info_states,
                    self._action_ph: actions,
                    self._rewards_to_go_ph: rewards_to_go,
                    self._old_log_probs_ph: old_log_probs,
                    self._fine_tune_mode_ph: True,
                    self._gae_ph: gae,
                })

            if self._fine_tune_learn_steps == 1:
                print("Entropy on first fine tune steps: ", entropy)
                
        for _ in range(5):
            value_loss, _ = self.session.run(
                [self.critic_loss, self._ppo_value_learn_step],
                feed_dict={
                    self._info_state_ph: info_states,
                    self._rewards_to_go_ph: rewards_to_go,
                    self._fine_tune_mode_ph: True
                })


        self.actor_loss_list.append(actor_loss)
        self.value_loss_list.append(value_loss)
        self.entropy_list.append(entropy)
        if (len(self.actor_loss_list) > 100 and self._fine_tune_learn_steps % 100 == 0):
            self.actor_loss_list = self.actor_loss_list[-100:]
            self.value_loss_list = self.value_loss_list[-100:]
            self.entropy_list = self.entropy_list[-100:]
            print("Mean PPO Actor + Value losses and entropy last 100 updates: ", sum(self.actor_loss_list) / len(self.actor_loss_list), sum(self.value_loss_list) / len(self.value_loss_list), sum(self.entropy_list) / len(self.entropy_list))
            print("Reward scaling std: ", self.reward_scaler.rs.std)
        self._replay_buffer.reset()
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
    
        # initialization_weights,
        # initialization_opt,
        self.session.run(
            tf.group(*[
                initialization_policy,
                initialization_value,
                initialization_ppo_value_opt,
                initialization_ppo_policy_opt
            ]))

    def cumsum(self, x, discount):
        vals = [None for _ in range(len(x))]
        curr = 0 
        for i in range(len(x) - 1, -1, -1):
            vals[i] = curr = x[i] + discount  * curr
        return vals

    def add_trajectory(self, trajectory, action_trajectory, override_symmetric=False):
        gae = [np.zeros(self.num_players) for _ in range(len(trajectory) - 1)]
        rewards_to_go = [np.zeros(self.num_players) for _ in range(len(trajectory) - 1)]

        scaled_rewards = [self.reward_scaler(np.array(timestep.rewards)) for timestep in trajectory[1:]]  # the first timestep has None value 

        for p in range(self.num_players):
            observations = [timestep.observations["info_state"][p] for timestep in trajectory]
            vals = self.session.run([self.values], feed_dict={self._info_state_ph:observations})[0]
            vals = np.array(vals)

            rewards = np.reshape(np.array([joint_reward[p] for joint_reward in scaled_rewards]), (-1, 1))

            assert rewards.shape == vals[1:].shape
            deltas = rewards + self.discount_factor * vals[1:] - vals[:-1]

            curr_gae = self.cumsum(deltas, self.discount_factor * .95)  # .95 is lambda
            assert len(gae) == len(curr_gae)
            for i, val in enumerate(gae):
                val[p] = curr_gae[i]

            curr_rewards = self.cumsum(rewards, self.discount_factor)
            assert len(rewards_to_go) == len(curr_rewards)

            for i, val in enumerate(rewards_to_go):
                val[p] = curr_rewards[i]

        for i in range(len(trajectory) - 1):
            if trajectory[i].is_simultaneous_move():
                # NOTE: If is_symmetric, then add_transition will add observations/actions from BOTH players already
                # NOTE: Also insert action_trajectory[i+1]. If it is the last i, then we let action be 0 because it won't be used anyway
                next_action = action_trajectory[i+1] if (i+1) < len(action_trajectory) else [0 for _ in range(self.num_players)] 
                self.add_transition(trajectory[i], action_trajectory[i], trajectory[i+1], next_action, ret=rewards_to_go[i], gae=gae[i], override_symmetric=override_symmetric) 
            elif player == trajectory[i].observations["current_player"]:

                # TODO: Figure out this later...

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

                    curr_policy.add_transition(trajectory[i], action, trajectory[next_player_timestep_index], next_action, ret=rewards_to_go[i], override_symmetric=override_symmetric) 

    def add_transition(self, prev_time_step, prev_action, time_step, action, ret, gae, override_symmetric=False):
        """Adds the new transition using `time_step` to the replay buffer.

        Adds the transition from `self._prev_timestep` to `time_step` by
        `self._prev_action`.

        Args:
          prev_time_step: prev ts, an instance of rl_environment.TimeStep.
          prev_action: int, action taken at `prev_time_step`.
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

                store_action = prev_action[p] if not isinstance(prev_action, int) else prev_action 
                store_next_action = action[p] if not isinstance(action, int) else action

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