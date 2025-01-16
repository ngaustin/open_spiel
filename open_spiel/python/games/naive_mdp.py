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

"""Python implementation of simple fire extinguisher game.

This is primarily here to demonstrate simultaneous-move games in Python.
"""

import enum

import numpy as np

import pyspiel

_DEFAULT_PARAMS = {"max_game_length": 10}
_NUM_STATES = 9


_GAME_TYPE = pyspiel.GameType(
    short_name="naive_mdp",
    long_name="MDP for Ex2PSRO Proof-of-Concept",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS)


class BargainingGeneralizedGame(pyspiel.Game):
  """The game, from which states and observers can be made."""

  # pylint:disable=dangerous-default-value
  def __init__(self, params=_DEFAULT_PARAMS):
    self._max_game_length = params["max_game_length"]

    super().__init__(
        _GAME_TYPE,
        pyspiel.GameInfo(
            num_distinct_actions=2,
            max_chance_outcomes=2,
            num_players=2,
            min_utility=-20,
            max_utility=50,
            utility_sum=0.0, # this is not true but serves no purpose
            max_game_length=self._max_game_length), params)
  
  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return BargainingGeneralizedGameState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return BargainingGeneralizedGameObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)

class BargainingGeneralizedGameState(pyspiel.State):
  """Current state of the game."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)

    self._curr_state = 1
    self._action_to_state_change = {0: -1, 1: 1}
    self._reward_function = {tuple([0, 0]): None,               tuple([0, 1]): np.array([1, 1]),
                             tuple([1, 0]): np.array([1, 1]),   tuple([1, 1]): np.array([-1, -1]),
                             tuple([2, 0]): np.array([-1, -1]), tuple([2, 1]): np.array([-2, -2]),
                             tuple([3, 0]): np.array([-2, -2]), tuple([3, 1]): np.array([-1, -1]),
                             tuple([4, 0]): np.array([-1, -1]), tuple([4, 1]): np.array([-1, -1]),
                             tuple([5, 0]): np.array([-1, -1]), tuple([5, 1]): np.array([ 1,  1]),
                             tuple([6, 0]): np.array([ 1,  1]), tuple([6, 1]): np.array([ 3,  3]),
                             tuple([7, 0]): np.array([ 3,  3]), tuple([7, 1]): np.array([ 5,  5]),
                             tuple([8, 0]): np.array([ 5,  5]), tuple([8, 1]): None               }  # Maps a tuple of (state, action) to length-2 vector reward. 

    self._rewards = np.zeros(2)
    self._returns = np.zeros(2)
    self._turns_so_far = 0

    # For symmetry, make the first player random
    self._curr_player = np.random.choice([0, 1])
    self._game_over = False
    self._is_chance = False
    self._returns = np.zeros(2)

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    elif self._is_chance:
      return pyspiel.PlayerId.CHANCE
    else:
      return self._curr_player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    if self._curr_state == 0:
      return [1]
    if self._curr_state == _NUM_STATES - 1:
      return [0]
    return [0, 1]

  def _apply_action(self, action):
    """Applies the specified action (one player) to the state."""
    assert not self._is_chance 

    # TODO: Query the reward function given the current state and applied action
    self._rewards = np.copy(self._reward_function[tuple([self._curr_state, action])])

    # TODO: Update reward and return
    self._returns += self._rewards

    # TODO: Update the next state 
    self._curr_state += self._action_to_state_change[action]

    # TODO: Increment the turns so far
    self._turns_so_far += 1

    # TODO: Modify the current player 
    self._curr_player = (self._curr_player + 1) % 2

    # TODO: If the number of turns equals the max number of turns , self._game_over = True 
    if self._turns_so_far >= self.get_game().max_game_length():
      self._game_over = True 

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      raise Exception("Should not be a chance node")
    else:
      return str(self._curr_state)

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
    return ", ".join(
        self._action_to_string(pa.player, pa.action)
        for pa in self.full_history()
        if pa.player == player)


class BargainingGeneralizedGameObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  # This information is fed into observation.py. Then, some how, the observations for each player is retrieved through
  # here and passed into the respective algorithm. The self.dict is what matters it seems.

  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    assert not bool(params)
    self.iig_obs_type = iig_obs_type

    self.tensor = np.zeros(_NUM_STATES)
    self.dict = {"state": self.tensor}


  def set_from(self, state, player):
    self.tensor.fill(0)
    self.tensor[state._curr_state] = 1
    self.dict = {"state": self.tensor}

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    if self.iig_obs_type.public_info:
      return "state: {}".format(state._curr_state)
    else:
      return None
    

# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, BargainingGeneralizedGame)
