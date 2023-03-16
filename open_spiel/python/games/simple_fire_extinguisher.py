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

from datetime import datetime

_NUM_PLAYERS = 2
_DEFAULT_PARAMS = {"max_game_length": 40, "grid_size": 5}
_MAX_GRID_SIZE = 8  # we do this to keep the rewards scaled the same way between different grid sizes
# make it square because implementation depends on it
# _REWARDS = {"burn_reward": -.5, "extinguish_reward": 5, "collision_fire": -15}
_REWARDS = {"burn_reward": -.2, "extinguish_reward": 10, "collision_fire": -10}

max_distance = 2 * _DEFAULT_PARAMS["grid_size"]


class Unit(enum.IntEnum):
  empty = -1
  fire = -2
  player = 0


_GAME_TYPE = pyspiel.GameType(
    short_name="simple_fire_extinguisher",
    long_name="Simple fire extinguisher game",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True, # for test passing
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS)

class Action(enum.IntEnum):
  LEFT = 0
  RIGHT = 1
  UP = 2
  DOWN = 3


class SimpleFireExtinguisherGame(pyspiel.Game):
  """The game, from which states and observers can be made."""

  # pylint:disable=dangerous-default-value
  def __init__(self, params=_DEFAULT_PARAMS):
    max_game_length = params["max_game_length"]
    super().__init__(
        _GAME_TYPE,
        pyspiel.GameInfo(
            num_distinct_actions=4,
            max_chance_outcomes=2,
            num_players=2,
            min_utility=_REWARDS["burn_reward"] * _DEFAULT_PARAMS["max_game_length"] * (1.5 ** max_distance),
            max_utility=_REWARDS["extinguish_reward"],
            utility_sum=0.0,
            max_game_length=max_game_length), params)

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return SimpleFireExtinguisherGameState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return SimpleFireExtinguisherGameObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)

  def observation_tensor_size(self):
    return 2 * (_NUM_PLAYERS - 1) + 2 + 1  # location of other agents plus location of the fire

  def display_policies_in_context(self, policies):
    """ Given a list of policies, display information about them to the console."""
    # TODO: Implement this
    return

  def save_iteration_data(self, iteration_number, meta_probabilities, U, policies, save_folder_path):
      """ How to save the iteration data? """
      date_time_string = str(datetime.now())
      date_time_string = date_time_string.replace(':', '_')
      save_data_path = save_folder_path + date_time_string + "_" + "iteration_{}.npy".format(iteration_number)

      all_meta_probabilities = np.vstack(meta_probabilities)
      array_list = [all_meta_probabilities, np.stack(U, axis=0)]
      object_array_list = np.empty(2, object)
      object_array_list[:] = array_list

      with open(save_data_path, "wb") as npy_file:
          np.save(npy_file, object_array_list)
      return

  def max_welfare_for_trajectory(self):
      # Does not account for collisions...assumes best path has no chance of collision
      init_state = self.new_initial_state()
      agent_locs = init_state.agent_locations
      fire_loc = init_state.fire_location
      welfare = 0
      for i in range(len(agent_locs)):
        manhattan_distance = SimpleFireExtinguisherGameState.distance(agent_locs[i], fire_loc)
        welfare += sum((1.5 ** (_MAX_GRID_SIZE - np.array(list(range(1, manhattan_distance))) + 1)) * _REWARDS["burn_reward"])
        welfare += _REWARDS["extinguish_reward"]
      return welfare

  def create_policy_heat_maps(self, policies, labels):
    """ Creates heat maps of all the policies of labels (list of list of ints)
    """
    # TODO: This is not implemented anywhere. Not needed
    return


class SimpleFireExtinguisherGameState(pyspiel.State):
  """Current state of the game."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._current_iteration = 1
    self._is_chance = False
    self._game_over = False
    self._rewards = np.zeros(_NUM_PLAYERS)
    self._returns = np.zeros(_NUM_PLAYERS)

    self.grid = np.zeros((_DEFAULT_PARAMS["grid_size"], _DEFAULT_PARAMS["grid_size"])) + Unit.empty
    self.agent_locations = [[0, 0], [0, 0]]
    self.fire_location = [_DEFAULT_PARAMS["grid_size"] - 1, _DEFAULT_PARAMS["grid_size"] - 1]
    self.fire_extinguished = False
    self._reached_max_iterations = False

    for i, loc in enumerate(self.agent_locations):
        self.grid[loc[0], loc[1]] = i
    self.grid[self.fire_location[0], self.fire_location[1]] = Unit.fire

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
    return [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]

  def _apply_actions(self, actions):
    """Applies the specified actions (per player) to the state."""
    assert not self._is_chance # and not self._game_over

    # For each of the actions, change the state of the player
    # Calculate the reward for each move
    # Allow players to occupy the same spot

    change_in_position = {Action.LEFT: (-1, 0), Action.RIGHT: (1, 0), Action.UP: (0, 1), Action.DOWN: (0, -1)}
    for i, a in enumerate(actions):
        curr_loc = self.agent_locations[i]
        delta = change_in_position[a]
        intended_next_pos = np.array([curr_loc[0] + delta[0], curr_loc[1] + delta[1]])
        intended_next_pos = np.clip(intended_next_pos, a_min=0, a_max=_DEFAULT_PARAMS["grid_size"] - 1)

        next_pos_status = self.grid[intended_next_pos[0], intended_next_pos[1]]

        if next_pos_status == Unit.fire:   # There is a person or fire there. So, no change in position and collision
            self._rewards[i] = _REWARDS["collision_fire"] + self._calculate_burn_reward(curr_loc)
        else:  # It is empty or has another player there. So, move there (or stay in place if the clip took care of it)
            # If there are no other players at the previous spot
            if not any([loc[0] == curr_loc[0] and loc[1] == curr_loc[1] for k, loc in enumerate(self.agent_locations) if k != i]):
                self.grid[curr_loc[0], curr_loc[1]] = Unit.empty
            self.agent_locations[i] = intended_next_pos
            self.grid[intended_next_pos[0], intended_next_pos[1]] = Unit.player
            self._rewards[i] = self._calculate_burn_reward(intended_next_pos)

    # Now, check if both agents are next to the fire. If they are, then the fire is extinguished
    if not self.fire_extinguished:
        all_agents_adjacent_to_fire = all([SimpleFireExtinguisherGameState.distance(loc, self.fire_location) == 1 for loc in self.agent_locations])
        self.fire_extinguished = all_agents_adjacent_to_fire
        if self.fire_extinguished:
            # print("FIRE EXTINGUISHED SUCCESSFULLY AT  ITERATION {} with positions {}".format(self._current_iteration, self.agent_locations))
            for i in range(len(actions)):
                self._rewards[i] += _REWARDS["extinguish_reward"]
            # TODO: set self._game_over to True here
            # self._game_over = True
        self.grid[self.fire_location[0], self.fire_location[1]] = Unit.empty



    self._returns += self._rewards
    self._current_iteration += 1
    # TODO: This ruins our Q value calculations...
    self._game_over = self._current_iteration > self.get_game().max_game_length() or self.fire_extinguished
    # self._reached_max_iterations = self._current_iteration > self.get_game().max_game_length()

  def _calculate_burn_reward(self, location):
      if self.fire_extinguished:
          return 0
      dist = SimpleFireExtinguisherGameState.distance(location, self.fire_location)
      assert (_MAX_GRID_SIZE - dist) >= 0 and _REWARDS["burn_reward"] < 0
      return 1.5 ** (_MAX_GRID_SIZE - dist + 1) * _REWARDS["burn_reward"]

  @staticmethod
  def distance(location1, location2):
      # Manhattan distance for ease of calculation and analysis
      # return np.sqrt((location1[0] - location2[0]) ** 2 + (location1[1] - location2[1]) ** 2)
      return abs(location1[0] - location2[0]) + abs(location1[1] - location2[1])

  def information_state_string(self, p):
    assert p == 0 or p == 1
    return self.information_state_string

  def information_state_string(self):
      # TODO: Change this to be more distinct based on what is said in spiel.h more identifiable!
      return "Iteration {}.".format(self._current_iteration)

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      raise Exception("Should not be a chance node")
    else:
      return Action(action).name

  # TODO: Add a different method called _hit_max_iterations(self): that tells rl_environment whether we have exceeded max iterations
  # TODO: Differentiate between is_terminal so that we know how to calculate the Q value

  def reached_max_iterations(self):
      return self._reached_max_iterations

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


class SimpleFireExtinguisherGameObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  # This information is fed into observation.py. Then, some how, the observations for each player is retrieved through
  # here and passed into the respective algorithm. The self.dict is what matters it seems.

  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    assert not bool(params)
    self.iig_obs_type = iig_obs_type
    # self.tensor = np.ones(1)  # This is just to indicate to RL algorithms that there is a single observation for both players given a state
    self.tensor = np.ones(4)
    self.dict = {"observation": self.tensor}

  def set_from(self, state, player):
    # self.tensor = np.ones(1) # This is just to indicate to RL algorithms that there is a single observation for both players given a state

    curr_player_location = state.agent_locations[player]
    agent_locations = state.agent_locations
    fire_location = state.fire_location
    obs = []
    for i, other_loc in enumerate(agent_locations):
        if i == player:
            continue
        loc_other_player = [other_loc[0] - curr_player_location[0], other_loc[1] - curr_player_location[1]]
        obs.extend(loc_other_player)
        # location_relative_to_fire = [other_loc[0] - fire_location[0], other_loc[1] - fire_location[1]]
        # near_fire = (abs(location_relative_to_fire[0]) + abs(location_relative_to_fire[1])) <= 2
        # loc_other_player = near_fire
        # obs.append(loc_other_player)
    loc_fire = [fire_location[0] - curr_player_location[0], fire_location[1] - curr_player_location[1]]
    obs.extend(loc_fire)
    obs.append(state.fire_extinguished)

    self.tensor = np.array(obs)

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

pyspiel.register_game(_GAME_TYPE, SimpleFireExtinguisherGame)
