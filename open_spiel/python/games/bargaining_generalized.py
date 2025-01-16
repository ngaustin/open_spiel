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
import itertools 
import numpy as np

import pyspiel

_SET_BARGAINING_PARAMS = {"item_types": 3, "min_total_valuation": 5, "max_total_valuation": 10, 
                          "min_pool_size": 5, "max_pool_size": 7, "max_game_length": 10}
_VARIABLE_BARGAINING_PARAMS = {"discount": .99}


_GAME_TYPE = pyspiel.GameType(
    short_name="bargaining_generalized",
    long_name="Bargaining for division of resources generalized game",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.SAMPLED_STOCHASTIC,  # not true but helps pass build/tests
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=2,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=False,
    parameter_specification=_VARIABLE_BARGAINING_PARAMS)


class BargainingGeneralizedGame(pyspiel.Game):
  """The game, from which states and observers can be made."""

  # pylint:disable=dangerous-default-value
  def __init__(self, params=_VARIABLE_BARGAINING_PARAMS):
    self.max_num_offers = _SET_BARGAINING_PARAMS["max_game_length"]
    self.discount = params["discount"]
    self.item_types = _SET_BARGAINING_PARAMS["item_types"]
    self.total_valuation = tuple([_SET_BARGAINING_PARAMS["min_total_valuation"], _SET_BARGAINING_PARAMS["max_total_valuation"]])
    self.pool_size = tuple([_SET_BARGAINING_PARAMS["min_pool_size"], _SET_BARGAINING_PARAMS["max_pool_size"]])
    
    num_offers, self.offer_index_to_division = self.get_offer_map(_SET_BARGAINING_PARAMS["item_types"], _SET_BARGAINING_PARAMS["min_pool_size"], _SET_BARGAINING_PARAMS["max_pool_size"])
    self._accept_index = num_offers

    super().__init__(
        _GAME_TYPE,
        pyspiel.GameInfo(
            num_distinct_actions=num_offers+1,
            max_chance_outcomes=0,
            num_players=2,
            min_utility=0,
            max_utility=_SET_BARGAINING_PARAMS["max_total_valuation"] * (_SET_BARGAINING_PARAMS["max_pool_size"] - (_SET_BARGAINING_PARAMS["item_types"] - 1)),
            utility_sum=0.0, # this is not true but serves no purpose
            max_game_length=self.max_num_offers), params)

  def get_offer_map(self, item_types, min_pool_size, max_pool_size):
    # Our sampling GUARANTEES that there is at least 1 of each item in the pool AND both players value all items at least > 0 
    # This implies that an item and have a count of [1, (pool_size - (item_types - 1))] (inclusive) in the pool 

    # Calculate the number of possible offers under the different pool sizes 
    # Then, create a hash map that maps OFFER_INDEX to a length-(item_types) offer, corresponding to what the offering player wants to take
    offers_found, offer_index_to_division = 0, {}


    # You only need to consider max_pool_size because if it's a plausible offer with the max number of items, it is also a plausible offer under a smaller pool size
    # These are offers, so we could still have a player taking only 0 of an item. However, the upper bound still applies.
    cartesian_product = itertools.product(list(range(0, max_pool_size - item_types + 2)), repeat=item_types)
    for candidate_offer in cartesian_product:
      # Conditions: for an offer to be plausible: 1) each item count must be <= its max item count and 2) the total number of items <= total pool size
      # (1) is accounted for in the cartesian product. (2) can be checked manually
      is_plausible_offer = sum(candidate_offer) <= max_pool_size
      if is_plausible_offer: 
        offer_index_to_division[offers_found] = candidate_offer
        offers_found += 1
    return offers_found, offer_index_to_division

  def get_accept_vector(self):
    return -np.ones(self.item_types)
  
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

    self._num_offers_so_far = 0
    self._all_offers_so_far = []

    # Sample valuations using numpy to pass test for RNG 
    # Ensure that all items get at least value of 1 AND that the sum of values falls within the range defined by self.get_game().total_valuation
    # Symmetry in valuations is ensured between players
    self._valuations = []
    min_value, max_value = self.get_game().total_valuation
    item_types = self.get_game().item_types
    for player in range(2):
      valid_valuation = False 
      while not valid_valuation:
        candidate = np.random.uniform(low=1.0, high=max_value-(item_types - 1), size=item_types)
        total_valuation = np.sum(candidate)
        valid_valuation = total_valuation >= min_value and total_valuation <= max_value
      self._valuations.append(candidate)

    # Sample pool using numpy to pass test for RNG 
    # Ensure there is at least 1 of each item AND that the sum of all items falls within the range defined by self.get_game().pool_size
    min_pool, max_pool = self.get_game().pool_size 
    valid_pool = False 
    while not valid_pool:
      candidate = np.random.randint(low=1, high=max_pool - (item_types - 2), size=item_types)
      sum_items = np.sum(candidate)
      valid_pool = sum_items >= min_pool and sum_items <= max_pool
    self._pool = candidate 

    # Precompute the legal actions for this game instance
    self.offer_index_to_division = self.get_game().offer_index_to_division
    self.legal_offers = []
    for offer_index, offer_division in self.offer_index_to_division.items():
      if all([offer_count <= pool_count for offer_count, pool_count in zip(offer_division, self._pool)]):
        self.legal_offers.append(offer_index)

    # For symmetry, make the first player random
    self._curr_player = np.random.choice([0, 1])
    self._offering_player = None
    self._game_over = False
    self._agreement_reached = False 
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
    assert player >= 0
    if self._num_offers_so_far > 0:
      return self.legal_offers + [self.get_game()._accept_index]
    return self.legal_offers

  def _apply_action(self, action):
    """Applies the specified action (one player) to the state."""
    assert not self._is_chance 

    # If it's accept, then call terminal 
    if action == self.get_game()._accept_index:
      self._game_over = True 
      self._agreement_reached = True
      self._offering_player = (self._curr_player + 1) % 2
    else:
      # Increment the number of offers 
      self._num_offers_so_far += 1

      # Get the proposed counter offer
      division_proposed = self.offer_index_to_division[action]

      # Sanity check
      # if not all([count <= pool_count for count, pool_count in zip(division_proposed, self._pool)]):
      #   print("Proposal: ", division_proposed)
      #   print("ActioN; ", action)
      #   print("Pool: ", self._pool)
      #   print("offer_to_index: ", self.offer_index_to_division)
      #   print("legal actions: ", self.legal_offers)
      #   assert all([count <= pool_count for count, pool_count in zip(division_proposed, self._pool)])

      # Append to the list of offers so far
      self._all_offers_so_far.append(np.array(division_proposed))

      # Next player 
      self._curr_player = (self._curr_player + 1) % 2

      # If we've reached more than the max number of offers (meaning a player rejected the offer at max_game_length offer), then terminal 
      if self._num_offers_so_far > self.get_game().max_num_offers:
        self._game_over = True

  # def action_index_to_offer_representation(self, action_index):
  #   return self.get_game().get_accept_vector() if action_index == self.get_game()._accept_index else self.offer_index_to_division[action_index]

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      raise Exception("Should not be a chance node")
    else:
      return "ACCEPT" if action == self.get_game()._accept_index else str(self.offer_index_to_division[action])

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    if not self._agreement_reached:
      return np.zeros(2)

    # discount = self.get_game().discount ** (self._num_offers_so_far)

    offering_player = self._offering_player
    most_recent_offer = self._all_offers_so_far[-1]

    self._returns[offering_player] = sum([value * item_count for value, item_count in zip(self._valuations[offering_player], most_recent_offer)])

    leftover_pool_items = [pool_count - offer_count for pool_count, offer_count in zip(self._pool, most_recent_offer)]
    
    # Sanity check
    assert all([count >= 0 for count in leftover_pool_items])

    other_player = (offering_player + 1) % 2
    self._returns[other_player] = sum([value * item_count for value, item_count in zip(self._valuations[other_player], leftover_pool_items)])

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

    self.item_types = _SET_BARGAINING_PARAMS["item_types"]
    self.max_num_offers = _SET_BARGAINING_PARAMS["max_game_length"]

    pieces = [("agreement_reached", 1), ("num_offers_so_far", 1), ("pool", self.item_types), ("valuation", self.item_types)]

    if self.iig_obs_type.perfect_recall:
      # Information state 
      pieces.append(("history", (self.max_num_offers + 1) * self.item_types))
    else:
      # Observation 
      pieces.append(("history", (self.item_types)))

    info_state_size = 2 + (self.item_types * 2) + ((self.max_num_offers + 1) * self.item_types) 
    observation_size = 2 + (self.item_types * 2) + self.item_types

    if self.iig_obs_type.perfect_recall:
      # Information state 
      self.tensor = np.zeros(info_state_size)
    else:
      # Observation 
      self.tensor = np.zeros(observation_size)

    self.dict = {}
    index = 0
    for name, size in pieces:
      self.dict[name] = self.tensor[index:index+size].reshape((size, ))
      index += size 

  def set_from(self, state, player):
    self.tensor.fill(0)

    index = 0
    self.tensor[index: index+self.dict["agreement_reached"].shape[0]] = int(state._agreement_reached)
    index += self.dict["agreement_reached"].shape[0]

    self.tensor[index: index+self.dict["num_offers_so_far"].shape[0]] = state._num_offers_so_far
    index += self.dict["num_offers_so_far"].shape[0]

    self.tensor[index: index+self.dict["pool"].shape[0]] = state._pool 
    index += self.dict["pool"].shape[0]

    self.tensor[index: index+self.dict["valuation"].shape[0]] = state._valuations[player]
    index += self.dict["valuation"].shape[0]
    
    if len(state._all_offers_so_far) > 0:
      if self.iig_obs_type.perfect_recall:
        # Insert full history
        offers_so_far = np.concatenate(state._all_offers_so_far)
        history = np.pad(offers_so_far, (0, (self.max_num_offers + 1) * self.item_types - offers_so_far.shape[0]), 'constant')
      else:
        # Only insert the most recent offer 
        history = state._all_offers_so_far[-1]
      self.dict["history"] = history
      self.tensor[index: index+self.dict["history"].shape[0]] = history 
    else:
      length = (self.max_num_offers + 1) * self.item_types if self.iig_obs_type.perfect_recall else self.item_types
      self.dict["history"] = np.zeros(length)
      self.tensor[index: index+self.dict["history"].shape[0]] = np.zeros(length)
    # print("TensorAggregate: ", self.tensor)


  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = ["agreement?", str(state._agreement_reached), "  "]
    pieces.extend(["# offers:", str(state._num_offers_so_far), "  "])
    pieces.extend(["pool:", ','.join([str(count) for count in state._pool]), "  "])
    pieces.extend(["valuation:", ','.join([str(np.round(value, 2)) for value in state._valuations[player]]), "  "])
    pieces.extend(["history:", ','.join([str(count) for count in self.dict["history"]]), "  "])
    return ' '.join(pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, BargainingGeneralizedGame)
