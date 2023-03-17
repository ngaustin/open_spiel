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

from datetime import datetime

_NUM_PLAYERS = 2
_DEFAULT_PARAMS = {"path_length": 10, "initial_distance_from_box": 4, "max_game_length": 30}
_REWARDS = {"effort_cost": -2, "push_cost": -4, "box_move_reward": 6, "box_reach_goal": 8}

"""
Create a game with a horizontal path that can hold two agents in a column. On the left side of the horizontal path, 
there are the agents. In the middle, there is a box. The agents must individually approach the box, push it together, 
and make the box reach the end of the horizontal path. 

Constraints:
- Agents can only move horizontally
- The box can only be pushed if both agents are immediately to the left of the box and both push it simultaneously 

Incentives:
- Agents receive add a negative reward for every movement they make 
- Agents receive 0 reward for not moving 
- Agents who try to push the box without the other doing so receive add a big negative reward
- Agents who collectively push the box receive a positive reward
- Finishing the task gives agents a small positive reward and exits the game

Parameters:
- Observation space: agent's row? Both of the agent's rows? 
- Action space: backwards, forwards, stay still
- Rewards: negative effort, negative single push, positive collective pushing, positive finish]
- Length of the path
- Initial distance between agents and box
"""
_SOCIAL_WELFARE_OPTIMIZATION = False  # Not implemented

_GAME_TYPE = pyspiel.GameType(
    short_name="simple_box_pushing",
    long_name="Python Simple Box Pushing",
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


class Action(enum.IntEnum):
  backward = 0
  stay = 1
  forward = 2

class Unit(enum.IntEnum):
  empty = 0
  player = 1
  box = 2


class SimpleBoxPushingGame(pyspiel.Game):
  """The game, from which states and observers can be made."""

  # pylint:disable=dangerous-default-value
  def __init__(self, params=_DEFAULT_PARAMS):
    max_game_length = params["max_game_length"]
    super().__init__(
        _GAME_TYPE,
        pyspiel.GameInfo(
            num_distinct_actions=3,
            max_chance_outcomes=0,
            num_players=2,
            min_utility=np.min(list(_REWARDS.values())) * max_game_length,
            max_utility=np.max(list(_REWARDS.values())) * max_game_length,
            utility_sum=0.0,
            max_game_length=max_game_length), params)

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return SimpleBoxPushingState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return SimpleBoxPushingObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)

  def observation_tensor_size(self):
    return 2 # CHANGED IT so that we only look at in front and adjacent tiles _DEFAULT_PARAMS["path_length"] + 1# * _NUM_PLAYERS

  def display_policies_in_context(self, policies):
      """ Display two tracks:
                1. all players where they are side by side
                2. all players assuming they are NOT adjacent to each other
          Only display the most recently added strategy
      """
      def print_strings_to_console(actions, title):
          """ actions is a list of lists indexed by player giving all actions in spots up to box_location
              can be used to display both the adjacent and non-adjacent policies"""
          print("\n{}\n".format(title))
          length_up_to_box = len(actions[0])
          square_side_length = 6  # make an even number
          print("-" * square_side_length * length_up_to_box)

          for i, player_actions in enumerate(actions):
              if i != 0:
                  print("\n")
              track_string = [arrow_displays[action_index] + (" " * square_side_length) for action_index in player_actions]
              track_string = "".join(track_string) + "|BOX|"
              print(track_string)
          print("-" * square_side_length * length_up_to_box)

      arrow_displays = {0: "<-", 1: "--", 2: "->"}
      which_policies_to_display = [-1, -2]
      for k in which_policies_to_display:
          print("Displaying policy from {} step(s) ago \n".format(-1 * k))
          recent_policies = {agent_index: agent_policies[k] for agent_index, agent_policies in enumerate(policies)}

          """ Display the policies assuming the two agents are next to each other """
          # Assume players start at the left of the track index 0
          together_actions = [[-1, -1], [-1, -1]]
          state = self.new_initial_state()
          for i in range(len(recent_policies)):
              probs = recent_policies[i].action_probabilities(state, i)
              together_actions[i][0] = np.argmax([probs[0], probs[1], probs[2]])
              print("TOGETHER ACTIONS FOR AGENT {} START".format(i), [probs[0], probs[1], probs[2]])
          for _ in range(_DEFAULT_PARAMS["initial_distance_from_box"] - 1):  # Get them close to the box
              state.apply_actions([2, 2])
          for i in range(len(recent_policies)):
              probs = recent_policies[i].action_probabilities(state, i)
              together_actions[i][1] = np.argmax([probs[0], probs[1], probs[2]])
              print("TOGETHER ACTIONS FOR AGENT {} NEAR BOX".format(i), [probs[0], probs[1], probs[2]])
          print_strings_to_console(together_actions, "Together Actions")

          """Display the policies assuming the two agents are not next to each other """
          separate_actions = [[-1, -1], [-1, --1]]
          state = self.new_initial_state()
          for _ in range(_DEFAULT_PARAMS["initial_distance_from_box"] - 1):
              state.apply_actions([2, 1])  # move the first player to the box
          for i in range(len(recent_policies)):
              probs = recent_policies[i].action_probabilities(state, i)
              separate_actions[i][(i + 1) % 2] = np.argmax([probs[0], probs[1], probs[2]])
              print("SEPARATE ACTIONS FOR AGENT {} START".format(i), [probs[0], probs[1], probs[2]])
          for _ in range(_DEFAULT_PARAMS["initial_distance_from_box"] - 1):
              state.apply_actions([0, 2])
          for i in range(len(recent_policies)):
              probs = recent_policies[i].action_probabilities(state, i)
              separate_actions[i][i] = np.argmax([probs[0], probs[1], probs[2]])
              print("SEPARATE ACTIONS FOR AGENT {} NEAR BOX".format(i), [probs[0], probs[1], probs[2]])
          print_strings_to_console(separate_actions, "Separate Actions")
      return

  def save_iteration_data(self, iteration_number, meta_probabilities, U, policies, save_folder_path):
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
      return 2 * ((_DEFAULT_PARAMS["initial_distance_from_box"] - 1) * _REWARDS["effort_cost"] +
                  (_DEFAULT_PARAMS["path_length"] - _DEFAULT_PARAMS["initial_distance_from_box"] - 1) * (_REWARDS["box_move_reward"] + _REWARDS["push_cost"]) +
                  _REWARDS["box_reach_goal"] + _REWARDS["push_cost"])

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


class SimpleBoxPushingState(pyspiel.State):
  """Current state of the game."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._current_iteration = 1
    self._game_over = False
    self._rewards = np.zeros(_NUM_PLAYERS)
    self._returns = np.zeros(_NUM_PLAYERS)
    self._is_chance = False

    # Initialize the track with initial locations of players and box
    self._track = np.zeros((_NUM_PLAYERS, _DEFAULT_PARAMS["path_length"])) + Unit.empty
    for i in range(_NUM_PLAYERS):
        self._track[i, 0] = Unit.player
        self._track[i, _DEFAULT_PARAMS["initial_distance_from_box"]] = Unit.box
    self.agent_locations = np.zeros(_NUM_PLAYERS).astype(int)
    self.box_location = _DEFAULT_PARAMS["initial_distance_from_box"]

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
    return [Action.backward, Action.stay, Action.forward]

  def _apply_actions(self, actions):
    """Applies the specified actions (per player) to the state."""
    assert not self._is_chance  # and not self._game_over
    # Put this at the beginning in case success criteria is reached. If put at the end, this would overwrite it
    self._game_over = self._current_iteration > self.get_game().max_game_length()

    # Check if all actions are forward and if all players are right behind the box
    push_box = all([a == Action.forward for a in actions]) and \
               all([self.box_location - loc == 1 for loc in self.agent_locations])
    # Update the box location if true
    if push_box:
        if self.box_location == _DEFAULT_PARAMS["path_length"] - 1:
            self._game_over = True
            for i in range(_NUM_PLAYERS):
                self._rewards[i] = _REWARDS["box_reach_goal"] + _REWARDS["push_cost"]
        else:
            self._track[:, self.box_location+1] = Unit.box
            self._track[:, self.box_location] = Unit.empty
            self.box_location += 1

    # Update the track and rewards
    # TODO: Support game_over vs max_iterations here
    if not self._game_over:
        for i in range(_NUM_PLAYERS):
            a = actions[i]
            loc = self.agent_locations[i]
            if a == Action.backward:  # and loc > 0:
                if loc > 0:
                    self._track[i, loc-1] = Unit.player
                    self._track[i, loc] = Unit.empty
                    self.agent_locations[i] = loc-1
                self._rewards[i] = _REWARDS["effort_cost"]
            elif push_box: # If all agents are behind box and all actions are forward
                self._track[i, loc+1] = Unit.player
                self._track[i, loc] = Unit.empty
                self.agent_locations[i] = loc+1
                self._rewards[i] = _REWARDS["box_move_reward"] + _REWARDS["push_cost"]
            elif a == Action.forward:
                # Check if the agent is behind the box. If not, then move it forward. Otherwise, keep him where he is
                if self.box_location - loc > 1:
                    self._track[i, loc+1] = Unit.player
                    self._track[i, loc] = Unit.empty
                    self.agent_locations[i] = loc+1
                    self._rewards[i] = _REWARDS["effort_cost"]
                else:  # Tried to push the box but could not
                    self._rewards[i] = _REWARDS["push_cost"]
            else:
                assert a == Action.stay  # or (a == Action.backward and loc == 0)

    self._returns += self._rewards
    self._current_iteration += 1

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      raise Exception("Should not be a chance node")
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


class SimpleBoxPushingObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  # This information is fed into observation.py. Then, some how, the observations for each player is retrieved through
  # here and passed into the respective algorithm. The self.dict is what matters it seems.

  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    assert not bool(params)
    self.iig_obs_type = iig_obs_type
    # self.tensor = np.ones(1)  # This is just to indicate to RL algorithms that there is a single observation for both players given a state
    self.tensor = np.zeros(3) # np.ones(_DEFAULT_PARAMS["path_length"] + 1)# * _NUM_PLAYERS)
    self.dict = {"observation": self.tensor}

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    # Each agent gets two values: box in front (1 or -1) and player adjacent (1 or -1)
    assert player == 0 or player == 1
    agent_locations = state.agent_locations
    box_location = state.box_location
    adjacent_to_each_other = int(all([loc == agent_locations[0] for loc in agent_locations]))

    self.tensor = np.array([1 if (box_location - agent_locations[player] == 1) else -1,
                            1 if adjacent_to_each_other else -1])  # np.append(state._track[player], 1 if adjacent_to_each_other else 0)

    # self.tensor = np.array([1 if (box_location - agent_locations[player] == 1) else -1 for player in range(len(agent_locations))])

    # self.tensor = np.array([agent_locations[player], box_location, 1 if adjacent_to_each_other else -1])
    self.dict["observation"] = self.tensor

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    if self.iig_obs_type.public_info:
      return (f"us:{state.action_history_string(player)} "
              f"op:{state.action_history_string(1 - player)}")
    else:
      return None


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, SimpleBoxPushingGame)
