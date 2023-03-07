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
import random

from open_spiel.python import rl_agent
from open_spiel.python import simple_nets
from open_spiel.python.utils.replay_buffer import ReplayBuffer

# Temporarily disable TF2 behavior until code is updated.
tf.disable_v2_behavior()

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward ret")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9


class Imitation(rl_agent.AbstractAgent):
    """DQN Agent implementation in TensorFlow.

    See open_spiel/python/examples/breakthrough_dqn.py for an usage example.
    """

    def __init__(self,
                 player_id,
                 consensus_kwargs, 
                 num_actions, 
                 state_representation_size):
        """Initialize the DQN agent."""

        # This call to locals() is used to store every argument used to initialize
        # the class instance, so it can be copied with no hyperparameter change.
        self._kwargs = locals()
        self.session = consensus_kwargs["session"]

        self.player_id = player_id
        self._num_actions = num_actions
        self.state_representation_size = state_representation_size

        self.boltzmann = consensus_kwargs["boltzmann"]
        self.mode = consensus_kwargs["imitation_mode"]

        self.epochs = consensus_kwargs["training_epochs"]
        self.minimum_entropy = consensus_kwargs["minimum_entropy"]
        self.lr = consensus_kwargs["deep_network_lr"]
        self.batch = consensus_kwargs["batch_size"]
        self.hidden_layer_size = consensus_kwargs["hidden_layer_size"]
        self.n_hidden_layers = consensus_kwargs["n_hidden_layers"]

        self.layer_sizes = [self.hidden_layer_size] * self.n_hidden_layers

        # Initialize replay
        self.replay_buffer = ReplayBuffer(np.inf)

        # Initialize the FF network 
        self.net = simple_nets.MLP(self.state_representation_size,
                                   self.layer_sizes, self._num_actions)

        self._variables = self.net.variables[:]

        if self.mode == "prob_action":
            self._info_state_ph = tf.placeholder(
                shape=[None, self.state_representation_size],
                dtype=tf.float32,
                name="info_state_ph")
            self._action_ph = tf.placeholder(
                shape=[None], dtype=tf.int32, name="action_ph")
            
            self._return_ph = tf.placeholder(
                shape=[None], dtype=tf.float32, name="return_ph"
            )

            self.log_probs = self.net(self._info_state_ph)
            loss_class = tf.losses.softmax_cross_entropy 

            # Convert the actions to one hot vectors 
            self.one_hot_vectors = tf.one_hot(self._action_ph, depth=self._num_actions)

            # TODO: Consider making entropy regularizer so that the policy doesn't collapse?

            # Plug into cross entropy class 
            self._loss = tf.reduce_mean(loss_class(self.one_hot_vectors, self.log_probs, weights=self._return_ph))

        elif self.mode == "prob_reward":
            self._info_state_ph = tf.placeholder(
                shape=[None, state_representation_size],
                dtype=tf.float32,
                name="info_state_ph")
            self._action_ph = tf.placeholder(
                shape=[None], dtype=tf.int32, name="action_ph")
            self._reward_ph = tf.placeholder(
                shape=[None], dtype=tf.float32, name="reward_ph")

            self.reward_predictions = self.net(self._info_state_ph)
            loss_class = tf.losses.mean_squared_error

            # Gather the correct indices from the reward predictions 
            action_indices = tf.stack([tf.range(tf.shape(self.reward_predictions)[0]), self._action_ph], axis=-1)
            reward_predictions_selected = tf.gather_nd(self.reward_predictions, action_indices)

            # Make sure reward targets are tensor and in the right shape 
            targets = tf.convert_to_tensor(self._reward_ph)

            # Plug into MSE loss
            self._loss = tf.reduce_mean(loss_class(labels=targets, predictions=reward_predictions_selected))
        else:
            raise NotImplemented

        # Initialize Adam Optimizer 
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self._learn_step = self._optimizer.minimize(self._loss)
        self._initialize()

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

            info_state = np.reshape(info_state, [1, -1])
            if self.mode == "prob_action":
                # Run session to get logits. Then, softmax over them
                logits = self.session.run(self.log_probs, feed_dict={self._info_state_ph: info_state})[0]
                # TODO: Check if the entropy has collapsed here
            elif self.mode == "prob_reward":
                logits = self.session.run(self.reward_predictions, feed_dict={self._info-state_ph: info_state})[0]
            else:
                raise NotImplemented


            action_probs = np.exp(self.boltzmann * logits) / np.sum(np.exp(self.boltzmann * logits))

            # action_probs = action_probs.eval()  # to numpy...this is what causes the slow down

            action = np.random.choice(self._num_actions, p=action_probs)
            # print("{}, {}, {}, {}".format(after_get_info, after_get_logits, after_boltzmann, after_conversion))
            return rl_agent.StepOutput(action=action, probs=action_probs)
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

        o = prev_time_step.observations["info_state"][self.player_id][:]
        r = sum(time_step.rewards) # WELFARE
        prev_action = prev_action[self.player_id]

        transition = Transition(
            info_state=(
                prev_time_step.observations["info_state"][self.player_id][:]),
            action=prev_action,
            reward=r, 
            ret=ret)

        self.replay_buffer.add(transition)


    def learn(self):
        """Compute the loss on sampled transitions and perform a Q-network update.

        If there are not enough elements in the buffer, no loss is computed and
        `None` is returned instead.

        Returns:
          The average loss obtained on this batch of transitions or `None`.
        """

        length = len(self.replay_buffer)
        dataset = self.replay_buffer.sample(length)  # samples without replacement so take the entire thing. Random order
        indices = list(range(length))
        for ep in range(self.epochs):
            i, batches, loss_total = 0, 0, 0
            dataset = random.sample(dataset, len(dataset))
            while i < length:
                transitions = dataset[i: min(length, i+self.batch)] 
                info_states = [t.info_state for t in transitions]
                actions = [t.action for t in transitions]
                rewards = [t.reward for t in transitions]
                rets = [t.ret for t in transitions]

                if self.mode == "prob_action":
                    loss, _, log_probs, one_hots = self.session.run(
                    [self._loss, self._learn_step, self.log_probs, self.one_hot_vectors],
                    feed_dict={
                        self._info_state_ph: info_states,
                        self._action_ph: actions,
                        self._return_ph: rets,
                    })

                    # Just to track entropy
                    loss = self.session.run(
                        [self._loss],
                    feed_dict={
                        self._info_state_ph: info_states,
                        self._action_ph: actions,
                        self._return_ph: [1 for _ in rets],
                    })
                    # print("ACTIONS: {}".format(actions))
                    # print("LOG PROBS: {}, ONE HOTS: {}".format(log_probs, one_hots))
                    # print(yes)
                elif self.mode == "prob_reward":
                    loss, _ = self.session.run(
                        [self._loss, self._learn_step],
                        feed_dict={
                            self._info_state_ph: info_states,
                            self._action_ph: actions,
                            self._reward_ph: rewards,
                        })

                else:
                    raise NotImplemented 
                loss_total += loss
                i += self.batch
                batches +=1 
            if ep % 100 == 0:
                print("Average loss for epoch {}: {}".format(ep, loss_total / float(batches)))

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
