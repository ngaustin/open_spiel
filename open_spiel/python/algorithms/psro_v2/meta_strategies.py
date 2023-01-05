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

"""Meta-strategy solvers for PSRO."""

import numpy as np

from open_spiel.python.algorithms import lp_solver
from open_spiel.python.algorithms import projected_replicator_dynamics
from open_spiel.python.algorithms import regret_matching
import pyspiel


EPSILON_MIN_POSITIVE_PROBA = 1e-8


def uniform_strategy(solver, return_joint=False):
  """Returns a Random Uniform distribution on policies.

  Args:
    solver: GenPSROSolver instance.
    return_joint: If true, only returns marginals. Otherwise marginals as well
      as joint probabilities.

  Returns:
    uniform distribution on strategies.
  """
  policies = solver.get_policies()
  policy_lengths = [len(pol) for pol in policies]
  result = [np.ones(pol_len) / pol_len for pol_len in policy_lengths]
  if not return_joint:
    return result
  else:
    joint_strategies = get_joint_strategy_from_marginals(result)
    return result, joint_strategies


def softmax_on_range(number_policies):
  x = np.array(list(range(number_policies)))
  x = np.exp(x-x.max())
  x /= np.sum(x)
  return x


def uniform_biased_strategy(solver, return_joint=False):
  """Returns a Biased Random Uniform distribution on policies.

  The uniform distribution is biased to prioritize playing against more recent
  policies (Policies that were appended to the policy list later in training)
  instead of older ones.

  Args:
    solver: GenPSROSolver instance.
    return_joint: If true, only returns marginals. Otherwise marginals as well
      as joint probabilities.

  Returns:
    uniform distribution on strategies.
  """
  policies = solver.get_policies()
  if not isinstance(policies[0], list):
    policies = [policies]
  policy_lengths = [len(pol) for pol in policies]
  result = [softmax_on_range(pol_len) for pol_len in policy_lengths]
  if not return_joint:
    return result
  else:
    joint_strategies = get_joint_strategy_from_marginals(result)
    return result, joint_strategies


def renormalize(probabilities):
  """Replaces all negative entries with zeroes and normalizes the result.

  Args:
    probabilities: probability vector to renormalize. Has to be one-dimensional.

  Returns:
    Renormalized probabilities.
  """
  probabilities[probabilities < 0] = 0
  probabilities = probabilities / np.sum(probabilities)
  return probabilities


def get_joint_strategy_from_marginals(probabilities):
  """Returns a joint strategy matrix from a list of marginals.

  Args:
    probabilities: list of probabilities.

  Returns:
    A joint strategy from a list of marginals.
  """
  probas = []
  for i in range(len(probabilities)):
    probas_shapes = [1] * len(probabilities)
    probas_shapes[i] = -1
    probas.append(probabilities[i].reshape(*probas_shapes))
  result = np.product(probas)
  return result.reshape(-1)


def nash_strategy(solver, return_joint=False):
  """Returns nash distribution on meta game matrix.

  This method only works for two player zero-sum games.

  Args:
    solver: GenPSROSolver instance.
    return_joint: If true, only returns marginals. Otherwise marginals as well
      as joint probabilities.

  Returns:
    Nash distribution on strategies.
  """
  meta_games = solver.get_meta_game()
  if not isinstance(meta_games, list):
    meta_games = [meta_games, -meta_games]
  meta_games = [x.tolist() for x in meta_games]
  if len(meta_games) != 2:
    raise NotImplementedError(
        "nash_strategy solver works only for 2p zero-sum"
        "games, but was invoked for a {} player game".format(len(meta_games)))
  nash_prob_1, nash_prob_2, _, _ = (
      lp_solver.solve_zero_sum_matrix_game(
          pyspiel.create_matrix_game(*meta_games)))
  result = [
      renormalize(np.array(nash_prob_1).reshape(-1)),
      renormalize(np.array(nash_prob_2).reshape(-1))
  ]

  if not return_joint:
    return result
  else:
    joint_strategies = get_joint_strategy_from_marginals(result)
    return result, joint_strategies


def prd_strategy(solver, return_joint=False, initial_strategies=None):
  """Computes Projected Replicator Dynamics strategies.

  Args:
    solver: GenPSROSolver instance.
    return_joint: If true, only returns marginals. Otherwise marginals as well
      as joint probabilities.

  Returns:
    PRD-computed strategies.
  """
  meta_games = solver.get_meta_game()
  if not isinstance(meta_games, list):
    meta_games = [meta_games, -meta_games]
  kwargs = solver.get_kwargs()
  # TODO: Confirm that kwargs["prd_gamma"] is .01
  print("Solving PRD with minimum meta probability {} ... ".format(kwargs["prd_gamma"]))
  result = projected_replicator_dynamics.projected_replicator_dynamics(
      meta_games, prd_initial_strategies=initial_strategies, **kwargs)
  if not return_joint:
    return result
  else:
    joint_strategies = get_joint_strategy_from_marginals(result)
    return result, joint_strategies

def rrd_strategy(solver, return_joint=False, initial_strategies=None):
  return

def prd_collab_strategy(solver, regret_lambda=1, return_joint=False):
  boltzmann = 0
  # First index: player   Second index: strategy index
  meta_games = solver.get_meta_game()
  # consensus_returns = solver.get_consensus_returns()  # Assume it returns a list of lists representing
                                                      # the trajectory returns of each rollout
  kwargs = solver.get_kwargs()

  welfare = np.zeros(meta_games[0].shape)
  for i in range(len(meta_games)):
    welfare = welfare + meta_games[i]

  # Initial strategies are a list of numpy arrays where we prioritize the max of the U matrix assuming other player
  # Uses the same thing
  initial_strategies = []

  # for i in range(len(meta_games)):
  #   explore_policy_returns = consensus_returns[i]
  #   max_over_all_opponent_strategies = np.max(welfare, axis=(i+1) % 2) # TODO: THIS ASSUME A 2-PLAYER GAME
  #   if explore_policy_returns[-1] != None:
  #     max_over_all_opponent_strategies[-1] = max(max_over_all_opponent_strategies[-1], explore_policy_returns[-1])
  #   maxes_numerator = np.exp(boltzmann * np.max(welfare, axis=(i+1) % 2))
  #   strategy = (maxes_numerator / np.sum(maxes_numerator))
  #   initial_strategies.append(strategy)
  # print("Initial meta-strategies: ", initial_strategies)
  # result_from_prd = prd_strategy(solver, return_joint, initial_strategies=initial_strategies)  # This should be a 2d np array.
  result = projected_replicator_dynamics.regularized_replicator_dynamics(
      meta_games, regret_lambda=regret_lambda, **kwargs) # prd_initial_strategies=initial_strategies, **kwargs)

  if return_joint:
    print("THIS IS NOT SUPPORTED")
    assert False
    #joint_strategies_from_prd = result_from_prd[1]
    #result_from_prd = result_from_prd[0]

  # epsilon_explore = .7
  # epsilon_explore_decay = .8
  # num_iterations = solver._iterations


  """
  explore_mss = []
  for i in range(len(meta_games)): # for each player
    explore_policy_returns = consensus_returns[i]
    values_not_none = np.array([ret for ret in explore_policy_returns if ret is not None])
    # sum_exp = np.sum(np.exp(values_not_none))
    curr_meta_strategy = []
    for ret in explore_policy_returns:
      if ret is None:
        curr_meta_strategy.append(0)
      else:
        # curr_meta_strategy.append(np.exp(ret) / sum_exp)
        curr_meta_strategy.append(1.0 / len(values_not_none))  # TODO: Right now, this is uniform

    if sum(curr_meta_strategy) == 0:
      # Edge case typically in beginning of PSRO training where there are no exploration policies
      curr_meta_strategy = [1.0 / len(explore_policy_returns) for _ in range(len(explore_policy_returns))]
    explore_mss.append(np.array(curr_meta_strategy))
  """
  """
  # TODO: Calculate the max betweeen decayed and actual
  individual_welfares, initial_strategies = [], []
  for i in range(len(meta_games)):  # for each player
    welfare_copy = welfare.copy()
    curr_consensus_return = consensus_returns[i]
    for j, ret in enumerate(curr_consensus_return):
      if ret is not None:
        welfare_copy[j, j] = max(welfare_copy[j, j], (epsilon_consensus ** (num_iterations - 1)) * ret)  # assume 2-player game
    individual_welfares.append(welfare_copy)

    # TODO: Create the initial meta-strategies that make at least one of the values w/ max utility better than all others
    curr_initial_strategies = []
    for k in range(len(meta_games)):
      revised_meta_game = individual_welfares[i]
      maxes_numerator = np.exp(boltzmann * np.max(revised_meta_game, axis=k))
      strategy = (maxes_numerator / np.sum(maxes_numerator)).tolist()
      curr_initial_strategies.append(strategy)
    initial_strategies.append(curr_initial_strategies)

  print("Individual welfares: ", individual_welfares)
  print("Initial strategies", initial_strategies)

  # TODO: Use PRD to create two different strategies (unless the two edited U matrices are identical)
  explore_mss = []
  for i in range(len(meta_games)):
    curr_welfare = individual_welfares[i]
    all_welfares = [curr_welfare, curr_welfare]
    res = projected_replicator_dynamics.projected_replicator_dynamics(
          all_welfares, **solver.get_kwargs())
    explore_mss.append(res[i])
  explore_mss = np.array(explore_mss)
  """

  # probability_exploration = epsilon_explore * (epsilon_explore_decay ** (num_iterations - 1))
  # print(explore_mss[0], result_from_prd[0])
  # result = [probability_exploration * explore_mss[i] + (1 - probability_exploration) * result_from_prd[i] for i in range(len(explore_mss))]

  return result


def rm_strategy(solver, return_joint=False):
  """Computes regret-matching strategies.

  Args:
    solver: GenPSROSolver instance.
    return_joint: If true, only returns marginals. Otherwise marginals as well
      as joint probabilities.

  Returns:
    PRD-computed strategies.
  """
  meta_games = solver.get_meta_game()
  if not isinstance(meta_games, list):
    meta_games = [meta_games, -meta_games]
  kwargs = solver.get_kwargs()
  result = regret_matching.regret_matching(meta_games, **kwargs)
  if not return_joint:
    return result
  else:
    joint_strategies = get_joint_strategy_from_marginals(result)
    return result, joint_strategies


META_STRATEGY_METHODS = {
    "uniform_biased": uniform_biased_strategy,
    "uniform": uniform_strategy,
    "nash": nash_strategy,
    "prd": prd_strategy,
    "rm": rm_strategy,
    "prd_collab": prd_collab_strategy,
}
