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

"""Example running PSRO on OpenSpiel Sequential games.

To reproduce results from (Muller et al., "A Generalized Training Approach for
Multiagent Learning", ICLR 2020; https://arxiv.org/abs/1909.12823), run this
script with:
  - `game_name` in ['kuhn_poker', 'leduc_poker']
  - `n_players` in [2, 3, 4, 5]
  - `meta_strategy_method` in ['alpharank', 'uniform', 'nash', 'prd']
  - `rectifier` in ['', 'rectified']

The other parameters keeping their default values.
"""

import time
from datetime import datetime

from absl import app
from absl import flags
import numpy as np

# pylint: disable=g-bad-import-order
import pyspiel
import tensorflow.compat.v1 as tf
import sys
# pylint: enable=g-bad-import-order

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python.algorithms.psro_v2 import best_response_oracle
from open_spiel.python.algorithms.psro_v2 import psro_v2
from open_spiel.python.algorithms.psro_v2 import rl_oracle

from open_spiel.python.algorithms.psro_v2 import rl_oracle_cooperative
from open_spiel.python.algorithms.psro_v2 import rl_policy
from open_spiel.python.algorithms.psro_v2 import strategy_selectors
from open_spiel.python.algorithms.psro_v2 import utils
from open_spiel.python.algorithms import config

FLAGS = flags.FLAGS

# Game-related
flags.DEFINE_string("game_name", "kuhn_poker", "Game name.")
flags.DEFINE_integer("n_players", 2, "The number of players.")

# PSRO related
flags.DEFINE_string("meta_strategy_method", "alpharank",
                    "Name of meta strategy computation method.")
flags.DEFINE_integer("number_policies_selected", 1,  # CHANGED THIS from 5
                     "Number of new strategies trained at each PSRO iteration.")
flags.DEFINE_integer("sims_per_entry", 200, # 1000,  # CHANGED THIS from 1000
                     ("Number of simulations to run to estimate each element"
                      "of the game outcome matrix."))

flags.DEFINE_integer("gpsro_iterations", 100,
                     "Number of training steps for GPSRO.")
flags.DEFINE_bool("symmetric_game", False, "Whether to consider the current "
                  "game as a symmetric game.")

# General Cooperative Consensus Policy Stuff
flags.DEFINE_bool("consensus_imitation", False, "Whether to use consensus oracle as well.")
flags.DEFINE_string("consensus_oracle", "q_learn", "Choice of oracle for exploration policy. "
                                                      "Choices are trajectory, trajectory_deep, q_learn")
flags.DEFINE_bool("joint_action", False, "Whether to train in joint state space when fitting exploration Q learning")
flags.DEFINE_string("trajectory_mode", "prob_reward", "How to fit to a trajectory. Options are prob_reward and prob_action")
flags.DEFINE_integer("n_top_trajectories", 1, "Number of trajectories to take from each of the BR simulations")
flags.DEFINE_bool("rewards_joint", True, "Whether to select trajectories and optimize consensus policies on joint rewards")
flags.DEFINE_float("proportion_uniform_trajectories", 0, "Proportion of taken trajectories that will be uniformly sampled across non-high return ones")

# Reward Fitting
flags.DEFINE_float("boltzmann", 1.0, "Boltzmann constant for softmax when reward trajectory fitting in DQN")

# Deep Trajectory Fitting
flags.DEFINE_float("consensus_deep_network_lr", 3e-4, "Learning Rate when training network for trajectory/joint q learning")
flags.DEFINE_float("consensus_deep_policy_network_lr", 3e-4, "Separate learning rate for policy network in CQL ")
flags.DEFINE_integer("consensus_update_target_every", 1, "Update target network")
flags.DEFINE_integer("consensus_batch_size", 128, "Batch size when training consensus network offline")
flags.DEFINE_integer("consensus_hidden_layer_size", 50, "Hidden layer size for consensus network")
flags.DEFINE_integer("consensus_n_hidden_layers", 2, "Number of hidden layers in consensus network")
flags.DEFINE_integer("consensus_training_epochs", 1, "Number of training epochs for offline BC training")
flags.DEFINE_integer("consensus_training_steps", int(1e3), "Number of training steps for offline RL training")
flags.DEFINE_float("consensus_minimum_entropy", .8, "Entropy of policy required to stop or until epochs run out")
flags.DEFINE_float("consensus_tau", 1e-3, "Soft update for target q network")

# Deep CQL or R-BVE
flags.DEFINE_float("alpha", 5.0, "Hyperparameter for q value minimization")
flags.DEFINE_float("eta", .05, "Gap between difference in values for regularization")
flags.DEFINE_float("beta", .5, "Amount of weight put on difference between trajectory returns")
flags.DEFINE_integer("max_buffer_size_fine_tune", 100000, "Fine tuning buffer size")
flags.DEFINE_integer("min_buffer_size_fine_tune", 50000, "Minimum number of entries in buffer to fine tune")
flags.DEFINE_bool("fine_tune", False, "Determines whether to fine tune the consensus policy")
flags.DEFINE_bool("clear_trajectories", False, "Determines whether to clear the trajectory list after every iteration of PSRO")
flags.DEFINE_float("psi", 1.0, "How much probability fine tune in joint space")
flags.DEFINE_float("eps_clip", .2, "PPO epsilon boundary clip")
flags.DEFINE_float("ppo_entropy", .01, "PPO entropy regularization")
flags.DEFINE_integer("epochs_ppo", 80, "PPO epochs")
flags.DEFINE_integer("minibatches_ppo", 5, "PPO minibatches")
flags.DEFINE_float("policy_constraint", .1, "Policy constraint regularization in PPO fine tuning")
flags.DEFINE_float("sac_target_entropy", .9, "SAC target entropy")
flags.DEFINE_float('sac_alpha', .2, "SAC alpha value for entropy regularization")

# RRD and MSS
flags.DEFINE_float("regret_lambda_init", .7, "Lambda threshold for RRD initially")
flags.DEFINE_float("regret_lambda_final", 0, "Lambda threshold decay every iteration")
flags.DEFINE_integer("regret_steps", 20, "Number of PSRO iterations to take to decrease regret")
flags.DEFINE_float("minimum_exploration_init", 0, "Minimum amount of profile weight on exploration policies")
flags.DEFINE_float("final_exploration", 0, "After annealing, the minimum amount of profile weight on exploration policies")

# Saving Data Path
flags.DEFINE_string("save_folder_path",  "../examples/data/simple_box_pushing", "Where to save iteration data. Will save one file for each iteration in the given folder")

""" THIS IS TABULAR Q LEARN STUFF"""
flags.DEFINE_float("step_size", 1e-3, "Learning rate in tabular q learning")
flags.DEFINE_float("discount_factor", .99, "Gamma in RL learning")

# Rectify options
flags.DEFINE_string("rectifier", "",
                    "Which rectifier to use. Choices are '' "
                    "(No filtering), 'rectified' for rectified.")
flags.DEFINE_string("training_strategy_selector", "probabilistic",
                    "Which strategy selector to use. Choices are "
                    " - 'top_k_probabilities': select top "
                    "`number_policies_selected` strategies. "
                    " - 'probabilistic': Randomly samples "
                    "`number_policies_selected` strategies with probability "
                    "equal to their selection probabilities. "
                    " - 'uniform': Uniformly sample `number_policies_selected` "
                    "strategies. "
                    " - 'rectified': Select every non-zero-selection-"
                    "probability strategy available to each player.")

# General (RL) agent parameters
flags.DEFINE_integer("number_training_steps", int(1e6), "Number of environment "
                     "steps per RL policy. Used for PG, DQN, and Tabular Q")
flags.DEFINE_float("self_play_proportion", 0.0, "Self play proportion")
flags.DEFINE_integer("hidden_layer_size", 32, "Hidden layer size")  # CHANGED THIS
flags.DEFINE_integer("batch_size", 32, "Batch size")  # CHANGED FROM 32
flags.DEFINE_float("sigma", 0.0, "Policy copy noise (Gaussian Dropout term).")
flags.DEFINE_string("optimizer_str", "adam", "'adam' or 'sgd'")

# Policy Gradient Oracle related
flags.DEFINE_string("loss_str", "qpg", "Name of loss used for BR training.")
flags.DEFINE_integer("num_q_before_pi", 8, "# critic updates before Pi update")
flags.DEFINE_integer("n_hidden_layers", 3, "# of hidden layers")  # CHANGED THIS
flags.DEFINE_float("entropy_cost", 0.001, "Self play proportion")
flags.DEFINE_float("critic_learning_rate", 1e-2, "Critic learning rate")
flags.DEFINE_float("pi_learning_rate", 1e-3, "Policy learning rate.")

# DQN
flags.DEFINE_float("dqn_learning_rate", 1e-2, "DQN learning rate.")  # CHAGNED FROM 1e-2
flags.DEFINE_integer("update_target_network_every", 1000, "Update target "  # CHANGED FROM 1000
                     "network every [X] steps")
flags.DEFINE_integer("learn_every", 10, "Learn every [X] steps.")  # CHANGED FROM 10
flags.DEFINE_integer("min_buffer_size_to_learn", 1000, "Learn after getting certain number of transitions")
flags.DEFINE_integer("max_buffer_size", int(1e4), "Buffer Size")
flags.DEFINE_integer("epsilon_decay_duration", 1000, "Number of steps for epsilon from 1 to .1")

# General
flags.DEFINE_integer("seed", 1, "Seed.")
flags.DEFINE_bool("local_launch", False, "Launch locally or not.")
flags.DEFINE_bool("verbose", True, "Enables verbose printing and profiling.")

def save_iteration_data(iteration_number, meta_probabilities, U, save_folder_path, training_returns, ppo_training_data):
      """ How to save the iteration data? """
      date_time_string = str(datetime.now())
      date_time_string = date_time_string.replace(':', '_')
      save_data_path = save_folder_path + date_time_string + "_" + "iteration_{}.npy".format(iteration_number)

      all_meta_probabilities = np.vstack(meta_probabilities)
      array_list = [all_meta_probabilities, np.stack(U, axis=0), training_returns, ppo_training_data]
      object_array_list = np.empty(4, object)
      object_array_list[:] = array_list

      with open(save_data_path, "wb") as npy_file:
          np.save(npy_file, object_array_list)
      return


def init_dqn_responder(sess, env):
  """Initializes the Policy Gradient-based responder and agents."""
  state_representation_size = env.observation_spec()["info_state"][0]  # TODO: CHECK THIS IS NON-ZERO
  global_state_representation_size = env.observation_spec()["global_state"][0] if len(env.observation_spec()["global_state"]) > 0 else FLAGS.n_players * state_representation_size
  # print("Rep size: ", state_representation_size)
  num_actions = env.action_spec()["num_actions"]

  gpu_devices = tf.config.list_physical_devices('GPU')
  device = None

  if len(gpu_devices) > 0:
    print(" \n Found gpu device. Using in network passes \n ")
    assert tf.test.is_gpu_available()
    device = "/gpu:0"
  else:
    print(" \n Did not find gpu device. Using cpu. \n ")
    device = "/cpu:0"

  agent_class = rl_policy.DQNPolicy
  agent_kwargs = {
      "session": sess,
      "double": True,
      "state_representation_size": state_representation_size,
      "symmetric": FLAGS.symmetric_game,
      "num_actions": num_actions,
      "hidden_layers_sizes": [FLAGS.hidden_layer_size] * FLAGS.n_hidden_layers,
      "batch_size": FLAGS.batch_size,
      "learning_rate": FLAGS.dqn_learning_rate,
      "update_target_network_every": FLAGS.update_target_network_every,
      "learn_every": FLAGS.learn_every,
      "optimizer_str": FLAGS.optimizer_str,
      "min_buffer_size_to_learn": FLAGS.min_buffer_size_to_learn,
      "replay_buffer_capacity": FLAGS.max_buffer_size,
      "epsilon_decay_duration": FLAGS.epsilon_decay_duration,
      "discount_factor": FLAGS.discount_factor
  }

  consensus_kwargs={
    "session": sess,
    "device": device,
    "state_representation_size": global_state_representation_size if FLAGS.joint_action else state_representation_size,
    "alpha": FLAGS.alpha,
    "consensus_oracle":FLAGS.consensus_oracle,
    "consensus_imitation": FLAGS.consensus_imitation,
    "imitation_mode":FLAGS.trajectory_mode,
    "num_simulations_fit":FLAGS.n_top_trajectories,
    "proportion_uniform_trajectories":FLAGS.proportion_uniform_trajectories,
    "joint_action": FLAGS.joint_action,
    "rewards_joint": FLAGS.rewards_joint,
    "boltzmann": FLAGS.boltzmann,
    "training_epochs": FLAGS.consensus_training_epochs,
    "training_steps": FLAGS.consensus_training_steps,
    "update_target_every": FLAGS.consensus_update_target_every,
    "minimum_entropy":FLAGS.consensus_minimum_entropy,
    "deep_network_lr": FLAGS.consensus_deep_network_lr,
    "deep_policy_network_lr": FLAGS.consensus_deep_policy_network_lr,
    "batch_size": FLAGS.consensus_batch_size,
    "hidden_layer_size": FLAGS.consensus_hidden_layer_size,
    "n_hidden_layers": FLAGS.consensus_n_hidden_layers,
    "num_players": FLAGS.n_players,
    "symmetric": FLAGS.symmetric_game,
    "tau": FLAGS.consensus_tau,
    "discount": FLAGS.discount_factor,
    "eta": FLAGS.eta,
    "beta": FLAGS.beta,
    "max_buffer_size_fine_tune": FLAGS.max_buffer_size_fine_tune,
    "min_buffer_size_fine_tune": FLAGS.min_buffer_size_fine_tune,
    "fine_tune": FLAGS.fine_tune,
    "clear_trajectories": FLAGS.clear_trajectories,
    "psi": FLAGS.psi,
    "eps_clip": FLAGS.eps_clip,
    "ppo_entropy_regularization": FLAGS.ppo_entropy,
    "policy_constraint": FLAGS.policy_constraint,
    "epochs_ppo": FLAGS.epochs_ppo,
    "minibatches_ppo": FLAGS.minibatches_ppo,
    "sac_target_entropy": FLAGS.sac_target_entropy,
    "sac_alpha": FLAGS.sac_alpha,
    "consensus_imitation": FLAGS.consensus_imitation
  }

  print("Agent Arguments: ")
  for key, value in sorted(agent_kwargs.items()):
    print("{}: {}".format(key, value))
  print('\n')

  print("Consensus Arguments: ")
  for key, value in sorted(consensus_kwargs.items()):
    print("{}: {}".format(key, value))
  print('\n')

  # if FLAGS.consensus_imitation:
  oracle = rl_oracle_cooperative.RLOracleCooperative(
      env,
      agent_class,
      agent_kwargs,
      consensus_kwargs=consensus_kwargs,
      number_training_steps=FLAGS.number_training_steps,
      self_play_proportion=FLAGS.self_play_proportion,
      sigma=FLAGS.sigma)
  """
  else:
      oracle = rl_oracle.RLOracle(
          env,
          agent_class,
          agent_kwargs,
          number_training_steps=FLAGS.number_training_steps,
          self_play_proportion=FLAGS.self_play_proportion,
          sigma=FLAGS.sigma)"""

  agents = [
      agent_class(  # pylint: disable=g-complex-comprehension
          env,
          player_id,
          **agent_kwargs)
      for player_id in range(FLAGS.n_players)
  ]
  for agent in agents:
    agent.freeze()
  return oracle, agents

def print_policy_analysis(policies, game, verbose=False):
  """Function printing policy diversity within game's known policies.

  Warning : only works with deterministic policies.
  Args:
    policies: List of list of policies (One list per game player)
    game: OpenSpiel game object.
    verbose: Whether to print policy diversity information. (True : print)

  Returns:
    List of list of unique policies (One list per player)
  """
  states_dict = get_all_states.get_all_states(game, np.infty, False, False)
  unique_policies = []
  for player in range(len(policies)):
    cur_policies = policies[player]
    cur_set = set()
    for pol in cur_policies:
      cur_str = ""
      for state_str in states_dict:
        if states_dict[state_str].current_player() == player:
          pol_action_dict = pol(states_dict[state_str])
          max_prob = max(list(pol_action_dict.values()))
          max_prob_actions = [
              a for a in pol_action_dict if pol_action_dict[a] == max_prob
          ]
          cur_str += "__" + state_str
          for a in max_prob_actions:
            cur_str += "-" + str(a)
      cur_set.add(cur_str)
    unique_policies.append(cur_set)
  if verbose:
    print("\n=====================================\nPolicy Diversity :")
    for player, cur_set in enumerate(unique_policies):
      print("Player {} : {} unique policies.".format(player, len(cur_set)))
  print("")
  return unique_policies


def gpsro_looper(env, oracle, agents):
  """Initializes and executes the GPSRO training loop."""
  sample_from_marginals = True  # TODO(somidshafiei) set False for alpharank
  training_strategy_selector = FLAGS.training_strategy_selector or strategy_selectors.probabilistic
  training_returns = []
  g_psro_solver = psro_v2.PSROSolver(
      env.game,
      oracle,
      initial_policies=agents,
      training_strategy_selector=training_strategy_selector,
      rectifier=FLAGS.rectifier,
      sims_per_entry=FLAGS.sims_per_entry,
      number_policies_selected=FLAGS.number_policies_selected,
      meta_strategy_method=FLAGS.meta_strategy_method,
      prd_iterations=50000,
      prd_gamma=1e-4,  # TODO: CHANGED THIS
      sample_from_marginals=sample_from_marginals,
      regret_lambda=FLAGS.regret_lambda_init,
      explore_mss=FLAGS.minimum_exploration_init,
      consensus_imitation=FLAGS.consensus_imitation,
      symmetric_game=FLAGS.symmetric_game)

  start_time = time.time()
  utils.display_meta_game(g_psro_solver.get_meta_game())
  for gpsro_iteration in range(FLAGS.gpsro_iterations):
    if FLAGS.verbose:
      print("Iteration : {}".format(gpsro_iteration))
      print("Time so far: {}".format(time.time() - start_time))

    g_psro_solver.iteration()
    if FLAGS.consensus_imitation:
      training_returns = oracle.get_training_returns()
    meta_game = g_psro_solver.get_meta_game()
    meta_probabilities = g_psro_solver.get_meta_strategies()
    policies = g_psro_solver.get_policies()

    g_psro_solver.update_regret_threshold(FLAGS.regret_lambda_final, FLAGS.regret_steps - gpsro_iteration - 1)
    g_psro_solver.update_explore_threshold(FLAGS.final_exploration, FLAGS.regret_steps - gpsro_iteration - 1)

    if FLAGS.verbose:
      utils.display_meta_game(meta_game)
      print("Metagame probabilities: ")
      for i, arr in enumerate(meta_probabilities):
          line = [str(np.round(prob, 4)) + "  " for prob in arr]
          print("Player #{}: ".format(i) + ''.join(line))
      if FLAGS.meta_strategy_method =='mgce':
          joint = g_psro_solver.get_joint_meta_probabilities()
          print("Joint meta strategy: ", joint)
      # env.game.display_policies_in_context(policies)

      save_folder_path = FLAGS.save_folder_path if FLAGS.save_folder_path[-1] == "/" else FLAGS.save_folder_path + "/"
      save_iteration_data(gpsro_iteration, meta_probabilities, meta_game, save_folder_path, training_returns, config.ppo_training_data)

    # The following lines only work for sequential games for the moment.
    """
    if env.game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL:
      aggregator = policy_aggregator.PolicyAggregator(env.game)
      aggr_policies = aggregator.aggregate(
          range(FLAGS.n_players), policies, meta_probabilities)

      exploitabilities, expl_per_player = exploitability.nash_conv(
          env.game, aggr_policies, return_only_nash_conv=False)

      _ = print_policy_analysis(policies, env.game, FLAGS.verbose)
      if FLAGS.verbose:
        print("Exploitabilities : {}".format(exploitabilities))
        print("Exploitabilities per player : {}".format(expl_per_player))
    """

def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  np.random.seed(FLAGS.seed)

  sys.setrecursionlimit(1500)

  if FLAGS.game_name=="harvest":
    game = pyspiel.load_game(FLAGS.game_name, {"rng_seed": FLAGS.seed})
  elif FLAGS.game_name=="bargaining":
    game = pyspiel.load_game(FLAGS.game_name, {"discount": 1.0})
  else:
    game = pyspiel.load_game(FLAGS.game_name)  # The iterated prisoners dilemma does not have "players" info type

  env = rl_environment.Environment(game, observation_type=rl_environment.ObservationType.OBSERVATION)

  import os
  num_cpus = os.cpu_count()
  print("Num cpu cores: ", num_cpus)
  os.environ["OMP_NUM_THREADS"] = "16"
  session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)

  # Initialize oracle and agents
  with tf.Session(config=session_conf) as sess:
    oracle, agents = init_dqn_responder(sess, env)
    """
    if FLAGS.oracle_type == "DQN":
      oracle, agents = init_dqn_responder(sess, env)
    elif FLAGS.oracle_type == "PG":
      oracle, agents = init_pg_responder(sess, env)
    elif FLAGS.oracle_type == "BR":
      oracle, agents = init_br_responder(env)
    elif FLAGS.oracle_type == "TAB_Q":
      oracle, agents = init_tabular_q_responder(sess, env)"""
    sess.run(tf.global_variables_initializer())
    gpsro_looper(env, oracle, agents)

if __name__ == "__main__":
  app.run(main)
