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

"""Python implementation of harvest game.

This is primarily here to demonstrate simultaneous-move games in Python.
"""

import enum

import numpy as np

import pyspiel

from datetime import datetime

_NUM_PLAYERS = 2
_DEFAULT_PARAMS = {"max_game_length": 1000, "view_size": 5, "apple_radius": 2}
_SPAWN_PROB = [0, .005, .02, .05] 
_REWARDS = {"apple": 1}

_HARVEST_MAP = [
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
    "W P   P      A    P AAAAA    P  A P  W",
    "W  P     A P AA    P    AAA    A  A  W",
    "W     A AAA  AAA    A    A AA AAAA   W",
    "W A  AAA A    A  A AAA  A  A   A A   W",
    "WAAA  A A    A  AAA A  AAA        A PW",
    "W A A  AAA  AAA  A A    A AA   AA AA W",
    "W  A A  AAA    A A  AAA    AAA  A    W",
    "W   AAA  A      AAA  A    AAAA       W",
    "W P  A       A  A AAA    A  A      P W",
    "WA  AAA  A  A  AAA A    AAAA     P   W",
    "W    A A   AAA  A A      A AA   A  P W",
    "W     AAA   A A  AAA      AA   AAA P W",
    "W A    A     AAA  A  P          A    W",
    "W       P     A         P  P P     P W",
    "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW",
]


# Original map but small subset. 98 apples spawn points. 14 x 24 of walking space
_HARVEST_MAP = [
    "WWWWWWWWWWWWWWWWWWWWWWWWWW",
    "W P   P      A   P  A P  W",
    "W  P     A P AA    A  A  W",
    "W     A AAA  AAA   AAA   W",
    "W A  AAA A    A    A A   W",
    "WAAA  A A    A  A A   A PW",
    "W A A  AAA  AAA    AA AA W",
    "W  A A  AAA    A  A A    W",
    "W   AAA  A      A AA     W",
    "W P  A       A  A  AAA P W",
    "WA  AAA  A  A  AA A A   PW",
    "W    A A   AAA  A  A   P W",
    "W     AAA   A A   AAA  P W",
    "W A    A     AAA   A  P  W",
    "W P     P     A          W",
    "WWWWWWWWWWWWWWWWWWWWWWWWWW",
]


# Original map but even smaller subset 14 x 14  62 apple spawn points

_HARVEST_MAP = [
    "WWWWWWWWWWWWWWWW",
    "W P   P     PAPW",
    "W  P     A P AAW",
    "W     A AAA  AAW",
    "W A  AAA A    AW",
    "WAAA  A A    A W",
    "W A A  AAA  AAAW",
    "W  A A  AAA    W",
    "W   AAA  A     W",
    "W    A       A W",
    "WA  AAA  A  A  W",
    "W    A A   AAAPW",
    "WP    AAA   A AW",
    "W A    A    PAAW",
    "W P     P    PAW",
    "WWWWWWWWWWWWWWWW",
]



# Small version
_HARVEST_MAP = [
  "WWWWWWWWWWWWWWWWWWW",
  "WPP               W",
  "W         A       W",
  "W        AAA      W",
  "W       AAAAA     W",
  "W        AAA      W",
  "W         A       W",
  "W                 W",
  "WWWWWWWWWWWWWWWWWWW",
]

# Smallest Version...set view to 5x5

_HARVEST_MAP = [
  "WWWWWWWWW",
  "WP AAA PW",
  "W AAAAA W",
  "WP AAA PW", 
  "WWWWWWWWW",
]





class Unit(enum.IntEnum):
  empty = -1
  apple = -2
  wall = -3
  # Players are denoted by their agent ID 0, 1, 2, 3...

_GAME_TYPE = pyspiel.GameType(
    short_name="harvest",
    long_name="Apple harvesting game social dilemma",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
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
  UP = 1
  RIGHT = 2
  DOWN = 3


class HarvestGame(pyspiel.Game):
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
            min_utility=0,
            max_utility=max_game_length * _REWARDS["apple"],
            utility_sum=0.0,
            max_game_length=max_game_length), params)

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return HarvestGameState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return HarvestGameObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)

  def observation_tensor_size(self):
    return _DEFAULT_PARAMS["view_size"] * _DEFAULT_PARAMS["view_size"] * 4  # assume it is a square view 

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
      return "not relevant because environment is stochastic."

  def create_policy_heat_maps(self, policies, labels):
    """ Creates heat maps of all the policies of labels (list of list of ints)
    """
    # TODO: This is not implemented anywhere. Not needed
    return


class HarvestGameState(pyspiel.State):
  """Current state of the game."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._current_iteration = 1
    self._is_chance = False
    self._game_over = False
    self._rewards = np.zeros(_NUM_PLAYERS)
    self._returns = np.zeros(_NUM_PLAYERS)

    # Define grid and locations
    x_max = len(_HARVEST_MAP)
    y_max = len(_HARVEST_MAP[0])
    self.grid = np.zeros((x_max, y_max)) + Unit.empty
    self.grid_shape = (x_max, y_max)
    self.agent_locations = []
    self.agent_spawn_points = []
    # self.apple_locations = []
    self.apple_spawn_points = []

    # Spawn apples in initially random locations
    for i in range(x_max):
      for j in range(y_max):
        if _HARVEST_MAP[i][j] == "A":
          self.grid[i, j] = Unit.apple
          # self.apple_locations.append((i, j)) # Make it a tuple so it is immutable
          self.apple_spawn_points.append((i, j))
        if _HARVEST_MAP[i][j] == "P":
          self.agent_spawn_points.append((i, j))
        if _HARVEST_MAP[i][j] == "W":
          self.grid[i, j] = Unit.wall
    # print("Resulting grid: ", self.grid)

    # Spawn players in random locations
    for i in range(_NUM_PLAYERS):
      spawn_choice = np.random.choice(len(self.agent_spawn_points))
      loc = self.agent_spawn_points.pop(spawn_choice)
      # loc = self.spawn_empty_location()
      self.grid[loc[0], loc[1]] = i
      self.agent_locations.append([loc[0], loc[1]])
    
    # TODO: Create a mask for detecting apples
    manhattan_distances = np.zeros((_DEFAULT_PARAMS["apple_radius"] * 2 + 1, _DEFAULT_PARAMS["apple_radius"] * 2 + 1))
    for i in range(manhattan_distances.shape[0]):
      for j in range(manhattan_distances.shape[1]):
        manhattan_distances[i, j] = abs(i - _DEFAULT_PARAMS["apple_radius"]) + abs(j - _DEFAULT_PARAMS["apple_radius"])
    
    self.mask = np.where(manhattan_distances <= _DEFAULT_PARAMS["apple_radius"], 1, 0)

    self._reached_max_iterations = False

  def spawn_empty_location(self):
    max_iterations = 1000
    iter = 0
    while iter < max_iterations:
      x = np.random.randint(low=0, high=self.grid_shape[0])
      y = np.random.randint(low=0, high=self.grid_shape[1])
      valid_place = self.grid[x, y] == Unit.empty
      if valid_place:
        return (x, y)
      location = (x, y)
      iter += 1

    # If could not find a random valid place, iterate through the entire grid to find somwhere empty. 
    for i in range(_DEFAULT_PARAMS["grid_shape"][0]):
      for j in range(_DEFAULT_PARAMS["grid_shape"][1]):
        if self.grid[i, j] == Unit.empty:
          return (i, j)

    raise Exception("Could not find an empty spawn location. The map is filled with agents or apples already! ")

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
    return [Action.LEFT, Action.UP, Action.RIGHT, Action.DOWN]

  def _spawn_apples(self):
    # Spawn apples probabilistically given the current configuration of apples

    random_numbers = np.random.uniform(size=len(self.apple_spawn_points))
    r = 0
    apple_radius = _DEFAULT_PARAMS["apple_radius"]
    
    for i in range(len(self.apple_spawn_points)):
      row, col = self.apple_spawn_points[i]

      if self.grid[row, col] == Unit.empty:
        num_apples = 0
        # This is the parallelized version for determining apple spawn probability using Manhattan distance
        x_min = row - apple_radius 
        x_max = row + apple_radius

        y_min = col - apple_radius 
        y_max = col + apple_radius 

        mask_subset = self.mask[max(-x_min, 0): (self.mask.shape[0]) - max(x_max + 1 - self.grid.shape[0], 0), 
                                max(-y_min, 0): (self.mask.shape[1]) - max(y_max + 1 - self.grid.shape[1], 0)]

        grid_subset = self.grid[max(x_min, 0): min(x_max + 1, self.grid.shape[0]), 
                                max(y_min, 0): min(y_max + 1, self.grid.shape[1])]
                          
        apple_mask = np.where(grid_subset == Unit.apple, 1, 0)

        num_apples = np.sum(np.multiply(mask_subset, apple_mask))

        """
        for j in range(-apple_radius, apple_radius + 1):
          for k in range(-apple_radius, apple_radius + 1):
            if j ** 2 + k ** 2 <= apple_radius:
              x, y = self.apple_spawn_points[i]
              if (0 <= x + j < self.grid.shape[0]
                and self.grid.shape[1] > y + k >= 0):
                if self.grid[x + j, y + k] == Unit.apple:
                  num_apples += 1
        """
        spawn_prob = _SPAWN_PROB[min(num_apples, 3)]
        rand_num = random_numbers[r]
        r += 1
        if rand_num < spawn_prob:
          # print("Spawning apple because {} is less than spawn prob {}".format(rand_num, spawn_prob))
          self.grid[row, col] = Unit.apple


  def _show_apple_map(self):
    print("### MAP ### ")
    for i in range(self.grid_shape[0]):
      print(self.grid[i, :])
    print("### MAP ### ")

  def _apply_actions(self, actions):
    """Applies the specified actions (per player) to the state."""
    assert not self._is_chance # and not self._game_over

    change_in_position = {Action.LEFT: (-1, 0), Action.RIGHT: (1, 0), Action.UP: (0, 1), Action.DOWN: (0, -1)}

    self._rewards = np.zeros(_NUM_PLAYERS)  # reset the rewards so that not affected by previous steps
    
    # print(self._show_apple_map())

    # Get the intended next locations of each of the agents 
    intended_positions = []
    for i, a in enumerate(actions):
      curr_loc = self.agent_locations[i]
      delta = change_in_position[a]
      intended_next_pos = np.array([curr_loc[0] + delta[0], curr_loc[1] + delta[1]])
      intended_next_pos[0] = np.clip(intended_next_pos[0], a_min=0, a_max=self.grid.shape[0])
      intended_next_pos[1] = np.clip(intended_next_pos[1], a_min=0, a_max=self.grid.shape[1])
      intended_positions.append(intended_next_pos)

    # Then, check if there are any conflicts. If there are not, then move them. Otherwise, you NONE of the agents that have the conflict will move.
    for i, pos in enumerate(intended_positions):
      collide_agent = np.any([np.array_equal(pos, other_pos) for j, other_pos in enumerate(intended_positions) if j != i])
      collide_wall = self.grid[pos[0], pos[1]] == Unit.wall
      if not (collide_agent or collide_wall):  # move if no collision with another player or boundary. Else, don't
        curr_loc = self.agent_locations[i]
        self.grid[curr_loc[0], curr_loc[1]] = Unit.empty 

        if self.grid[pos[0], pos[1]] == Unit.apple:
          self._rewards[i] += _REWARDS["apple"]
        
        # print("Current Iteration{}: Agent {} to position {}. Got Apple: {}".format(self._current_iteration, i, pos, self.grid[pos[0], pos[1]] == Unit.apple))
        self.grid[pos[0], pos[1]] = i  # new location of player i 
        self.agent_locations[i] = pos

        

    self._returns += self._rewards
    self._spawn_apples()
    # print("Actions: {}, Intended Positions: {}, Rewards {}, Returns {}".format(actions, intended_positions, self._rewards, self._returns))
    self._current_iteration += 1

    # Also insert GAME OVER if all of the apples are gone! 
    apples_are_gone = not np.any(self.grid == Unit.apple)
    self._game_over = self._current_iteration > self.get_game().max_game_length() or apples_are_gone
    # self._reached_max_iterations = self._current_iteration > self.get_game().max_game_length()

  @staticmethod
  def distance(location1, location2):
      # Manhattan distance for ease of calculation and analysis
      # return np.sqrt((location1[0] - location2[0]) ** 2 + (location1[1] - location2[1]) ** 2)
      return abs(location1[0] - location2[0]) + abs(location1[1] - location2[1])

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      raise Exception("Should not be a chance node")
    else:
      return Action(action).name

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


class HarvestGameObserver:
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
    assert _DEFAULT_PARAMS["view_size"] % 2 == 1  # ensure it is odd for simplicity in calculation
    
    curr_player_location = state.agent_locations[player]
    square_view_delta = _DEFAULT_PARAMS["view_size"] // 2 

    # First the x coordinates 
    x_left = curr_player_location[0] - square_view_delta
    x_right = curr_player_location[0] + square_view_delta # remember needs to be inclusive on this. So eventually, +1 on it

    if x_left < 0:
      x_right += (-x_left)
      x_left = 0
    elif x_right > state.grid.shape[0] - 1:
      correction = x_right - (state.grid.shape[0] - 1)
      x_right -= correction 
      x_left -= correction 

    # Then the y coordinates 
    y_bottom = curr_player_location[1] - square_view_delta
    y_top = curr_player_location[1] + square_view_delta

    if y_bottom < 0:
      y_top += (-y_bottom)
      y_bottom = 0
    elif y_top > state.grid.shape[1] - 1:
      correction = y_top - (state.grid.shape[1] - 1) 
      y_top -= correction
      y_bottom -= correction 
    
    grid_subset = state.grid[x_left: x_right + 1, y_bottom: y_top + 1]

    wall_indicator = np.where(grid_subset == Unit.wall, 1, 0).flatten()
    apple_indicator = np.where(grid_subset == Unit.apple, 1, 0).flatten()
    nothing_indicator = np.where(grid_subset == Unit.empty, 1, 0).flatten()

    # agent_indicator = np.where(grid_subset >= 0, 1, 0).flatten()

    other_agents = np.where((grid_subset >= 0) & (grid_subset != player), -1, 0)
    curr_agent = np.where(grid_subset == player, 1, 0)
    agent_indicator = (other_agents + curr_agent).flatten()

    obs = np.concatenate([wall_indicator, agent_indicator, apple_indicator, nothing_indicator])
    # obs = grid_subset.flatten()

    self.tensor = obs

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

pyspiel.register_game(_GAME_TYPE, HarvestGame)
