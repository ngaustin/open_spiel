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

"""Python implementation of iterated prisoner's dilemma.

This is primarily here to demonstrate simultaneous-move games in Python.
"""

import enum

import numpy as np

import pyspiel

_NUM_PLAYERS = 2
_DEFAULT_PARAMS = {"termination_probability": 0.125, "max_game_length": 10}
# _PAYOFF = [[5, 0], [10, 1]]
_PAYOFF = [[3, 3, -2], [1, 3, 0], [0, -2, 5]]
_SINGLE_STATE = False
_SOCIAL_WELFARE_OPTIMIZATION = False

"""   
What the game looks like: 
      a     b     c
a    3,3   3,1   -2,0
b    1,3   3,3   0,-2
c   0,-2  -2,0   5,5

The best response to c is only c. However, PSRO may encourage best response to opt  for a or b because c has low expected
utility (both in social welfare and individual) unless consensus is reached with the other player.
"""


_GAME_TYPE = pyspiel.GameType(
    short_name="simple_iterated_game",
    long_name="Python Simple Iterated Game",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,  # for test passing
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS)


# class Action(enum.IntEnum):
#   COOPERATE = 0
#   DEFECT = 1

class Action(enum.IntEnum):
  A = 0
  B = 1
  C = 2


class Chance(enum.IntEnum):
  CONTINUE = 0
  STOP = 1


class SimpleIteratedGame(pyspiel.Game):
  """The game, from which states and observers can be made."""

  # pylint:disable=dangerous-default-value
  def __init__(self, params=_DEFAULT_PARAMS):
    max_game_length = params["max_game_length"]
    super().__init__(
        _GAME_TYPE,
        pyspiel.GameInfo(
            num_distinct_actions=3,  # 2,
            max_chance_outcomes=2,
            num_players=2,
            min_utility=np.min(_PAYOFF) * max_game_length,
            max_utility=np.max(_PAYOFF) * max_game_length,
            utility_sum=0.0,
            max_game_length=max_game_length), params)
    self._termination_probability = params["termination_probability"]

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return SimpleIteratedGameState(self, self._termination_probability)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return SimpleIteratedGameObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)

  def observation_tensor_size(self):  # This did solve the bug...meaning that this method is overwriting some parent class
    return 1 if _SINGLE_STATE else 3 # THis is a test

  def display_policies_in_context(self, policies):
    """ Given a list of policies, display information about them to the console."""
    for i, agent_policies in enumerate(policies):
        print("Player {} policy information: ".format(i))
        for j, policy in enumerate(agent_policies):
            state = self.new_initial_state()
            action_probs = policy.action_probabilities(state, i)
            print("     Policy number {}    Prob action 0: {:2.3f}       Prob action 1: {:2.3f}     Prob action 2:{:2.3f}".format(j, action_probs[0], action_probs[1], action_probs[2]))
            if not _SINGLE_STATE:
                for k in range(3):
                    actions = [k, k]
                    state.apply_actions(actions)
                    action_probs = policy.action_probabilities(state, i)
                    print("     Policy number {}    Prob action 0: {:2.3f}       Prob action 1: {:2.3f}     Prob action 2:{:2.3f}".format( j, action_probs[0], action_probs[1], action_probs[2]))
        print("\n")
    return

  def save_iteration_data(self, iteration_number, meta_probabilities, U, policies, save_folder_path):
      """ Save a list of numpy arrays representing:
            [num_players, S] shape matrix of meta_probabilities (where S is the number of policies so far)
            [num_players, num_states, num_actions, S] shape matrix representing policies
            [num_players, S, S] shape matrix representing utilities (U)
      """
      save_data_path = save_folder_path + "iteration_{}.npy".format(iteration_number)
      all_meta_probabilities = np.vstack(meta_probabilities)
      S = meta_probabilities[0].size
      policy_matrix = np.zeros((_NUM_PLAYERS, 1 if _SINGLE_STATE else 4, self.num_distinct_actions(), S))
      for i, agent_policies in enumerate(policies):
          for j, policy in enumerate(agent_policies):
              state = self.new_initial_state()
              action_probs = policy.action_probabilities(state, i)
              policy_matrix[i, 0, 0, j] = action_probs[0]
              policy_matrix[i, 0, 1, j] = action_probs[1]
              policy_matrix[i, 0, 2, j] = action_probs[2]
              if not _SINGLE_STATE:
                  for k in range(3):
                      actions = [k, k]
                      state.apply_actions(actions)
                      action_probs = policy.action_probabilities(state, i)
                      policy_matrix[i, k + 1, 0, j] = action_probs[0]
                      policy_matrix[i, k + 1, 1, j] = action_probs[1]
                      policy_matrix[i, k + 1, 2, j] = action_probs[2]

      print("Saved policy matrix of size: ", policy_matrix.shape)
      array_list = [all_meta_probabilities, policy_matrix, np.stack(U, axis=0)]
      object_array_list = np.empty(3, object)
      object_array_list[:] = array_list
      with open(save_data_path, "wb") as npy_file:
          np.save(npy_file, object_array_list)
      return

  def max_welfare_for_trajectory(self):
      return np.max(np.array(_PAYOFF)) * _DEFAULT_PARAMS["max_game_length"] * 2

  def create_policy_heat_maps(self, policies, labels):
    """ Creates heat maps of all the policies of labels (list of list of ints)
    """
    state = self.new_initial_state()

    for i, agent_policies in enumerate(policies):
        agent_labels = labels[i]
        policy_subset = [agent_policies[index] for index in agent_labels]

        policy_info = ["Policy Number {}".format(labels[j]) for j in agent_labels]
        action_info = ["Prob Action {}".format(j) for j in range(3)]

        matrix = np.zeros((1, 3))
        for j, policy in enumerate(agent_policies):
            action_probs = policy.action_probabilities(state, i)
            current_array = np.array([[action_probs[0], action_probs[1], action_probs[2]]])
            matrix = np.vstack((matrix, current_array))

        utils.display_heat_map(action_info, policy_info, matrix[1:, :], "Policy Heat Map for Player {}".format(i))


class SimpleIteratedGameState(pyspiel.State):
  """Current state of the game."""

  def __init__(self, game, termination_probability):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._current_iteration = 1
    self._termination_probability = termination_probability
    self._is_chance = False
    self._game_over = False
    self._rewards = np.zeros(_NUM_PLAYERS)
    self._returns = np.zeros(_NUM_PLAYERS)

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every simultaneous-move game with chance.

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    elif self._is_chance:
      return pyspiel.PlayerId.CHANCE
    else:
      return pyspiel.PlayerId.SIMULTANEOUS

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    # return [Action.COOPERATE, Action.DEFECT]
    return [Action.A, Action.B, Action.C]

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self._is_chance
    return [(Chance.CONTINUE, 1 - self._termination_probability),
            (Chance.STOP, self._termination_probability)]

  """
  def _apply_action(self, action):
    # Applies the specified action to the state.
    # This is not called at simultaneous-move states.
    # assert self._is_chance and not self._game_over
    assert not self._game_over 
    print('CHANCE HERE')
    self._current_iteration += 1
    self._is_chance = False
    self._game_over = (action == Chance.STOP)
    if self._current_iteration > self.get_game().max_game_length():
      self._game_over = True
"""

  def _apply_actions(self, actions):
    """Applies the specified actions (per player) to the state."""
    assert not self._is_chance # and not self._game_over
    # print('NOT CHANCE HERE', self._current_iteration, self.get_game().max_game_length(), self._game_over)
    # self._is_chance = True
    if _SOCIAL_WELFARE_OPTIMIZATION:
        # If we are looking at social welfare optimization solely, then we add the two rewards together and
        # Assign the summed reward to both players
        sum_rewards = _PAYOFF[actions[0]][actions[1]] + _PAYOFF[actions[1]][actions[0]]
        self._rewards[0] = sum_rewards
        self._rewards[1] = sum_rewards
    else:
        self._rewards[0] = _PAYOFF[actions[0]][actions[1]]
        self._rewards[1] = _PAYOFF[actions[1]][actions[0]]
    self._returns += self._rewards
    self._current_iteration += 1
    self._game_over = self._current_iteration > self.get_game().max_game_length()

  def information_state_string(self, p):
    assert p == 0 or p == 1
    return self.information_state_string

  def information_state_string(self):
      # TODO: Change this to be more distinct based on what is said in spiel.h more identifiable!
      return "Iteration {}.".format(self._current_iteration)

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return Chance(action).name
    else:
      return Action(action).name

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over

  def rewards(self):
    """Reward at the previous step."""
    return self._rewards

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return self._returns

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return (f"p0:{self.action_history_string(0)} "
            f"p1:{self.action_history_string(1)}")

  def action_history_string(self, player):
    return "".join(
        self._action_to_string(pa.player, pa.action)[0]
        for pa in self.full_history()
        if pa.player == player)


class SimpleIteratedGameObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  # This information is fed into observation.py. Then, some how, the observations for each player is retrieved through
  # here and passed into the respective algorithm. The self.dict is what matters it seems.

  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    assert not bool(params)
    self.iig_obs_type = iig_obs_type
    # self.tensor = np.ones(1)  # This is just to indicate to RL algorithms that there is a single observation for both players given a state
    self.tensor = np.ones(1 if _SINGLE_STATE else 3)
    self.dict = {"observation": self.tensor}

  def set_from(self, state, player):
    # self.tensor = np.ones(1) # This is just to indicate to RL algorithms that there is a single observation for both players given a state
    assert player == 0 or player == 1
    if not _SINGLE_STATE:
        history = state.full_history()
        # print(history, len(history), self.tensor)
        action_history = [(player_object.player, player_object.action) for player_object in history[-2:] if player_object.player == abs(player - 1)]

        self.tensor = np.ones(3)
        if len(action_history) > 0:
            last_opposing_action = action_history[0][1]
            self.tensor[last_opposing_action] = 0
            # print(len(history), self.tensor, last_opposing_action)
    self.dict["observation"] = self.tensor
    # print(self.tensor)

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    if self.iig_obs_type.public_info:
      return (f"us:{state.action_history_string(player)} "
              f"op:{state.action_history_string(1 - player)}")
    else:
      return None


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, SimpleIteratedGame)
