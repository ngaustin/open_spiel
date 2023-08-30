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

"""Abstract class for meta trainers (Generalized PSRO, RNR, ...)

Meta-algorithm with modular behaviour, allowing implementation of PSRO, RNR, and
other variations.
"""

import numpy as np
from open_spiel.python.algorithms.psro_v2 import meta_strategies
from open_spiel.python.algorithms.psro_v2 import strategy_selectors
from open_spiel.python.algorithms.psro_v2 import utils
from open_spiel.python.rl_environment import StepType, TimeStep
import time 

_DEFAULT_STRATEGY_SELECTION_METHOD = "probabilistic"
_DEFAULT_META_STRATEGY_METHOD = "prd"

def get_time_step(state):
    """Returns a `TimeStep` without updating the environment.

    Returns:
      A `TimeStep` namedtuple containing:
        observation: list of dicts containing one observations per player, each
          corresponding to `observation_spec()`.
        reward: list of rewards at this timestep, or None if step_type is
          `StepType.FIRST`.
        discount: list of discounts in the range [0, 1], or None if step_type is
          `StepType.FIRST`.
        step_type: A `StepType` value.
    """
    observations = {
        "info_state": [],
        "legal_actions": [],
        "current_player": [],
        "serialized_state": [], 
        "global_state": []
    }
    rewards = []
    step_type = StepType.LAST if state.is_terminal() else StepType.MID

    if not state.is_chance_node():
      cur_rewards = state.rewards()
    else:
      cur_rewards = [None for _ in range(state.num_players())]

    for player_id in range(state.num_players()):
      rewards.append(cur_rewards[player_id])
      observations["info_state"].append(
          state.observation_tensor(player_id))

      observations["legal_actions"].append(state.legal_actions(player_id))
    observations["current_player"] = state.current_player()

    discounts = [0. for _ in range(state.num_players())]

    # For gym environments
    if hasattr(state, "last_info"):
      observations["info"] = state.last_info

    return TimeStep(
        observations=observations,
        rewards=rewards,
        discounts=discounts,
        step_type=step_type)


def _process_string_or_callable(string_or_callable, dictionary):
  """Process a callable or a string representing a callable.

  Args:
    string_or_callable: Either a string or a callable
    dictionary: Dictionary of shape {string_reference: callable}

  Returns:
    string_or_callable if string_or_callable is a callable ; otherwise,
    dictionary[string_or_callable]

  Raises:
    NotImplementedError: If string_or_callable is of the wrong type, or has an
      unexpected value (Not present in dictionary).
  """
  if callable(string_or_callable):
    return string_or_callable

  try:
    return dictionary[string_or_callable]
  except KeyError as e:
    raise NotImplementedError("Input type / value not supported. Accepted types"
                              ": string, callable. Acceptable string values : "
                              "{}. Input provided : {}".format(
                                  list(dictionary.keys()),
                                  string_or_callable)) from e


def sample_episode(state, policies):
  """Samples an episode using policies, starting from state.

  Args:
    state: Pyspiel state representing the current state.
    policies: List of policy representing the policy executed by each player.

  Returns:
    The result of the call to returns() of the final state in the episode.
        Meant to be a win/loss integer.
  """

  start = time.time()
  timestep = get_time_step(state)
  if state.is_terminal():
    return np.array(state.returns(), dtype=np.float32), [timestep], [] # None, None#

  if state.is_simultaneous_node():
    actions = [None] * state.num_players()
    for player in range(state.num_players()):
      # calls python.policy.__call__() wheich calls rl_policy.action_probabilities which calls _policy.step (_policy is dqn, imitation etc.)
      state_policy = policies[player](state, player)  
      outcomes, probs = zip(*state_policy.items())
      actions[player] = utils.random_choice(outcomes, probs)
    state.apply_actions(actions)
    rets, later_timesteps, later_actions = sample_episode(state, policies)
    return rets, [timestep] + later_timesteps, [actions] + later_actions # None, None# 
    # return sample_episode(state, policies)

  # Not implemented
  if state.is_chance_node():
    outcomes, probs = zip(*state.chance_outcomes())
  else:
    player = state.current_player()
    state_policy = policies[player](state, player)
    outcomes, probs = zip(*state_policy.items())

  applied_action = utils.random_choice(outcomes, probs)
  state.apply_action(applied_action)
  rets, later_timesteps, later_actions = sample_episode(state, policies)
  # We use [applied_action] because we keep consistency that actions are list of actions for each player
  return rets, [timestep] + later_timesteps, [[applied_action]] + later_actions # None, None# 


class AbstractMetaTrainer(object):
  """Abstract class implementing meta trainers.

  If a trainer is something that computes a best response to given environment &
  agents, a meta trainer will compute which best responses to compute (Against
  what, how, etc)
  This class can support PBT, Hyperparameter Evolution, etc.
  """

  # pylint:disable=dangerous-default-value
  def __init__(self,
               game,
               oracle,
               initial_policies=None,
               meta_strategy_method=_DEFAULT_META_STRATEGY_METHOD,
               training_strategy_selector=_DEFAULT_STRATEGY_SELECTION_METHOD,
               symmetric_game=False,
               number_policies_selected=1,
               **kwargs):
    """Abstract Initialization for meta trainers.

    Args:
      game: A pyspiel game object.
      oracle: An oracle object, from an implementation of the AbstractOracle
        class.
      initial_policies: A list of initial policies, to set up a default for
        training. Resorts to tabular policies if not set.
      meta_strategy_method: String, or callable taking a MetaTrainer object and
        returning a list of meta strategies (One list entry per player).
        String value can be:
              - "uniform": Uniform distribution on policies.
              - "nash": Taking nash distribution. Only works for 2 player, 0-sum
                games.
              - "prd": Projected Replicator Dynamics, as described in Lanctot et
                Al.
      training_strategy_selector: A callable or a string. If a callable, takes
        as arguments: - An instance of `PSROSolver`, - a
          `number_policies_selected` integer. and returning a list of
          `num_players` lists of selected policies to train from.
        When a string, supported values are:
              - "top_k_probabilites": selects the first
                'number_policies_selected' policies with highest selection
                probabilities.
              - "probabilistic": randomly selects 'number_policies_selected'
                with probabilities determined by the meta strategies.
              - "exhaustive": selects every policy of every player.
              - "rectified": only selects strategies that have nonzero chance of
                being selected.
              - "uniform": randomly selects 'number_policies_selected' policies
                with uniform probabilities.
      symmetric_game: Whether to consider the current game as symmetric (True)
        game or not (False).
      number_policies_selected: Maximum number of new policies to train for each
        player at each PSRO iteration.
      **kwargs: kwargs for meta strategy computation and training strategy
        selection
    """
    self._iterations = 0
    self._game = game
    self._oracle = oracle
    self._num_players = self._game.num_players()

    self.symmetric_game = symmetric_game
    self._game_num_players = self._num_players
    self._num_players = 1 if symmetric_game else self._num_players

    self._number_policies_selected = number_policies_selected

    print("Using {} as strategy method.".format(meta_strategy_method))
    meta_strategy_method = _process_string_or_callable(
        meta_strategy_method, meta_strategies.META_STRATEGY_METHODS)

    print("Using {} as training strategy selector.".format(
        training_strategy_selector))
    self._training_strategy_selector = _process_string_or_callable(
        training_strategy_selector,
        strategy_selectors.TRAINING_STRATEGY_SELECTORS)

    self._meta_strategy_method = meta_strategy_method
    self._kwargs = kwargs

    self._initialize_policy(initial_policies)
    self._initialize_game_state()
    self.update_meta_strategies()

  def _initialize_policy(self, initial_policies):
    return NotImplementedError(
        "initialize_policy not implemented. Initial policies passed as"
        " arguments : {}".format(initial_policies))

  def _initialize_game_state(self):
    return NotImplementedError("initialize_game_state not implemented.")

  def iteration(self, seed=None):
    """Main trainer loop.

    Args:
      seed: Seed for random BR noise generation.
    """
    self._iterations += 1
    print('Approximating Best Response')
    self.update_agents()  # Generate new, Best Response agents via oracle.
    print('Updating Empirical Game')
    self.update_empirical_gamestate(seed=seed)  # Update gamestate matrix.
    print('Computing meta_strategies')
    self.update_meta_strategies()  # Compute meta strategy (e.g. Nash)

  def update_meta_strategies(self):
    self._meta_strategy_probabilities = self._meta_strategy_method(self)
    if self.symmetric_game:
      self._meta_strategy_probabilities = [self._meta_strategy_probabilities[0]]

  def update_agents(self):
    return NotImplementedError("update_agents not implemented.")

  def update_empirical_gamestate(self, seed=None):
    return NotImplementedError("update_empirical_gamestate not implemented."
                               " Seed passed as argument : {}".format(seed))

  def sample_episodes(self, policies, num_episodes):
    """Samples episodes and averages their returns.

    Args:
      policies: A list of policies representing the policies executed by each
        player.
      num_episodes: Number of episodes to execute to estimate average return of
        policies.

    Returns:
      Average episode return over num episodes.
    """
    totals = np.zeros(self._num_players)
    all_trajectories = []
    all_action_trajectories = []
    all_returns = []

    for pol in policies:
      pol._policy.clear_state_tracking()
    
    for ep in range(num_episodes):
      start = time.time()
      rets, trajectory, actions = sample_episode(self._game.new_initial_state(),
                                        policies)
      # print("Episode returns: ", rets)
      totals += rets.reshape(-1)
      all_returns.append(rets.reshape(-1))
      all_trajectories.append(trajectory)
      all_action_trajectories.append(actions)
      
    return totals / num_episodes, all_trajectories, all_action_trajectories, all_returns #None, None, None#

  def get_meta_strategies(self):
    """Returns the Nash Equilibrium distribution on meta game matrix."""
    meta_strategy_probabilities = self._meta_strategy_probabilities
    if self.symmetric_game:
      meta_strategy_probabilities = (self._game_num_players *
                                     meta_strategy_probabilities)
    return [np.copy(a) for a in meta_strategy_probabilities]

  def get_meta_game(self):
    """Returns the meta game matrix."""
    meta_games = self._meta_games
    return [np.copy(a) for a in meta_games]

  def get_policies(self):
    """Returns the players' policies."""
    policies = self._policies
    if self.symmetric_game:
      # Notice that the following line returns N references to the same policy
      # This might not be correct for certain applications.
      # E.g., a DQN BR oracle with player_id information
      policies = self._game_num_players * policies
    return policies

  def get_kwargs(self):
    return self._kwargs
