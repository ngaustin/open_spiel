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
import os

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
flags.DEFINE_string("consensus_oracle", "trajectory_deep", "Choice of oracle for exploration policy. "
                                                      "Choices are trajectory, trajectory_deep, q_learn")
flags.DEFINE_bool("joint_action", False, "Whether to train in joint state space when fitting exploration Q learning")
flags.DEFINE_string("trajectory_mode", "prob_action", "How to fit to a trajectory. Options are prob_reward and prob_action")
flags.DEFINE_integer("n_top_trajectories", 1, "Number of trajectories to take from each of the BR simulations")
flags.DEFINE_bool("rewards_joint", False, "Whether to select trajectories and optimize consensus policies on joint rewards")
flags.DEFINE_bool("perturb_all", False, "Whether to use policy constraints on ALL policies after first iteration")
flags.DEFINE_string("exploration_policy_type", "Ex2PSRO", "Type of policy to regularize towards. Options: Ex2PSRO, MaxWelfare, MinWelfare, Uniform, PreviousBR")

# Parameter search-related model-saving functionality
flags.DEFINE_bool("save_models", False, "Whether to save the policy network models after SAC training")
flags.DEFINE_string("save_model_path", "./", "Relative path to save the policy network models if save_models is set to True")
flags.DEFINE_bool("load_models", False, "Whether to load the networks that wer saved in save_model_path based on psro iteration")
flags.DEFINE_integer("num_iterations_load_only", 0, "Number of iterations where PSRO should simply load models and not train BR")
flags.DEFINE_bool("regret_calculation_mode", False, "Whether to read from several files consisting of empirical gamestates/profiles and generate true best responses to them")

# SAC parameters
flags.DEFINE_float("value_clip", .2, "Value function clipping for sac")
flags.DEFINE_float("alpha", .01, "Entropy temperature for sac")
flags.DEFINE_integer("sac_batch_size", 64, "Batch size for each sac update")
flags.DEFINE_integer("sac_update_every", 10, "Sac performs one update every sac_update_every env steps")


# PPO parameters
flags.DEFINE_integer("max_buffer_size_fine_tune", 100000, "Fine tuning buffer size")
flags.DEFINE_integer("min_buffer_size_fine_tune", 50000, "Minimum number of entries in buffer to fine tune")
flags.DEFINE_integer("number_training_steps", int(1e6), "Number of environment "
                     "steps per RL policy. Used for PG, DQN, and Tabular Q")
flags.DEFINE_bool("fine_tune", False, "Determines whether to fine tune the consensus policy")
flags.DEFINE_bool("clear_trajectories", False, "Determines whether to clear the trajectory list after every iteration of PSRO")
flags.DEFINE_float("eps_clip", .2, "PPO epsilon boundary clip")
flags.DEFINE_float("eps_clip_value", .2, "PPO epsilon boundary clip for value ")
flags.DEFINE_float("ppo_entropy", .01, "PPO entropy regularization")
flags.DEFINE_integer("epochs_ppo", 80, "PPO epochs")
flags.DEFINE_integer("minibatches_ppo", 5, "PPO minibatches")
flags.DEFINE_float("policy_constraint", .1, "Policy constraint regularization in PPO fine tuning")
flags.DEFINE_integer("policy_constraint_steps", 20, "Number of PSRO iterations to take to decrease regret")
flags.DEFINE_float("fine_tune_policy_lr", 3e-5, "policy lr")
flags.DEFINE_float("fine_tune_value_lr", 3e-4, "value lr")
flags.DEFINE_float("entropy_decay_duration", .9, "proportion of training steps to decay entropy regularization for")
flags.DEFINE_float("transfer_policy_minimum_entropy", 0, "Minimum policy entropy to transfer policies across PSRO iterations")
flags.DEFINE_bool("transfer_policy", True, "Determines whether to transfer policy across PSRO iterations")
flags.DEFINE_integer("recovery_window", 200, "Window to calculate the recovery for PPO PSRO")

# Both PPO and offline training parameters
flags.DEFINE_integer("consensus_hidden_layer_size", 50, "Hidden layer size for consensus network")
flags.DEFINE_integer("consensus_n_hidden_layers", 2, "Number of hidden layers in consensus network")

# BC Offline Training Parameters
flags.DEFINE_float("consensus_deep_network_lr", 3e-4, "Learning Rate when training network for trajectory/joint q learning")
flags.DEFINE_integer("consensus_batch_size", 128, "Batch size when training consensus network offline")
flags.DEFINE_integer("consensus_training_epochs", 1, "Number of training epochs for offline BC training")
flags.DEFINE_float("consensus_minimum_entropy", .8, "Entropy of policy required to stop or until epochs run out")

# R-BVE Offline training parameters (shouldn't be used)
flags.DEFINE_integer("consensus_update_target_every", 1, "Update target network")
flags.DEFINE_integer("consensus_training_steps", int(1e3), "Number of training steps for offline RL training")
# flags.DEFINE_float("alpha", 5.0, "Hyperparameter for q value minimization")
flags.DEFINE_float("eta", .05, "Gap between difference in values for regularization")
flags.DEFINE_float("beta", .5, "Amount of weight put on difference between trajectory returns")

# RRD and MSS
flags.DEFINE_float("regret_lambda_init", 5, "Lambda threshold for RRD initially")
flags.DEFINE_float("regret_lambda_final", 0, "Lambda threshold decay every iteration")
flags.DEFINE_float("minimum_exploration_init", 0, "Minimum amount of profile weight on exploration policies")
flags.DEFINE_float("final_exploration", 0, "After annealing, the minimum amount of profile weight on exploration policies")
flags.DEFINE_integer("regret_steps", 25, "How many iterations regret should be annealed for")

# DQN for regret calculation (not used anymore)
flags.DEFINE_float("dqn_learning_rate", 1e-2, "DQN learning rate.")  # CHAGNED FROM 1e-2
flags.DEFINE_integer("update_target_network_every", 1000, "Update target "  # CHANGED FROM 1000
                     "network every [X] steps")
flags.DEFINE_integer("learn_every", 10, "Learn every [X] steps.")  # CHANGED FROM 10
flags.DEFINE_integer("min_buffer_size_to_learn", 1000, "Learn after getting certain number of transitions")
flags.DEFINE_integer("max_buffer_size", int(1e4), "Buffer Size")
flags.DEFINE_integer("epsilon_decay_duration", 1000, "Number of steps for epsilon from 1 to .1")
flags.DEFINE_integer("pretrained_policy_steps", 500, "Number of steps to anneal the probability of using the pretrained policy from 1 to 0")
flags.DEFINE_integer("regret_calculation_steps", int(2e5), "Number of steps when estimating regret")
flags.DEFINE_integer("hidden_layer_size", 50, "Hidden layer size")  # CHANGED THIS
flags.DEFINE_integer("n_hidden_layers", 2, "# of hidden layers")  # CHANGED THIS
flags.DEFINE_integer("batch_size", 32, "Batch size")  # CHANGED FROM 32

# General RL used for everything
flags.DEFINE_float("discount_factor", .99, "Gamma in RL learning")

# Saving Data Path
flags.DEFINE_string("save_folder_path", None, "Where to save iteration data. Will save one file for each iteration in the given folder")

# Tabular Q-Learning (not used)
flags.DEFINE_float("step_size", 1e-3, "Learning rate in tabular q learning")

# Rectify options (not used)
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
              

# Not used
flags.DEFINE_float("self_play_proportion", 0.0, "Self play proportion")
flags.DEFINE_float("sigma", 0.0, "Policy copy noise (Gaussian Dropout term).")
flags.DEFINE_string("optimizer_str", "adam", "'adam' or 'sgd'")
flags.DEFINE_string("loss_str", "qpg", "Name of loss used for BR training.")
flags.DEFINE_integer("num_q_before_pi", 8, "# critic updates before Pi update")
flags.DEFINE_float("entropy_cost", 0.001, "Self play proportion")
flags.DEFINE_float("critic_learning_rate", 1e-2, "Critic learning rate")
flags.DEFINE_float("pi_learning_rate", 1e-3, "Policy learning rate.")

# General
flags.DEFINE_integer("seed", 1, "Seed.")
flags.DEFINE_bool("local_launch", False, "Launch locally or not.")
flags.DEFINE_bool("verbose", True, "Enables verbose printing and profiling.")

def save_iteration_data(iteration_number, meta_probabilities, U, save_folder_path, training_returns, train_regret_returns, ppo_training_data, pure_br_returns):
      """ How to save the iteration data? """
      date_time_string = str(datetime.now())
      date_time_string = date_time_string.replace(':', '_')
      save_data_path = save_folder_path + date_time_string + "_" + "iteration_{}.npy".format(iteration_number)

      pathExists = os.path.exists(save_folder_path)
      if not pathExists:
        os.makedirs(save_folder_path)

      all_meta_probabilities = np.vstack(meta_probabilities)
      array_list = [all_meta_probabilities, np.stack(U, axis=0), training_returns, train_regret_returns, ppo_training_data, pure_br_returns]
      object_array_list = np.empty(len(array_list), object)
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
    "joint_action": FLAGS.joint_action,
    "rewards_joint": FLAGS.rewards_joint,
    "training_epochs": FLAGS.consensus_training_epochs,
    "training_steps": FLAGS.consensus_training_steps,
    "update_target_every": FLAGS.consensus_update_target_every,
    "minimum_entropy":FLAGS.consensus_minimum_entropy,
    "deep_network_lr": FLAGS.consensus_deep_network_lr,
    "batch_size": FLAGS.consensus_batch_size,
    "hidden_layer_size": FLAGS.consensus_hidden_layer_size,
    "n_hidden_layers": FLAGS.consensus_n_hidden_layers,
    "num_players": FLAGS.n_players,
    "symmetric": FLAGS.symmetric_game,
    "discount": FLAGS.discount_factor,
    "eta": FLAGS.eta,
    "beta": FLAGS.beta,
    "max_buffer_size_fine_tune": FLAGS.max_buffer_size_fine_tune,
    "min_buffer_size_fine_tune": FLAGS.min_buffer_size_fine_tune,
    "fine_tune": FLAGS.fine_tune,
    "clear_trajectories": FLAGS.clear_trajectories,
    "eps_clip": FLAGS.eps_clip,
    "eps_clip_value": FLAGS.eps_clip_value,
    "ppo_entropy_regularization": FLAGS.ppo_entropy,
    "policy_constraint": FLAGS.policy_constraint,
    "epochs_ppo": FLAGS.epochs_ppo,
    "minibatches_ppo": FLAGS.minibatches_ppo,
    "policy_constraint_steps": FLAGS.policy_constraint_steps,
    "perturb_all": FLAGS.perturb_all,
    "steps_fine_tune": FLAGS.number_training_steps,
    "fine_tune_policy_lr": FLAGS.fine_tune_policy_lr, 
    "fine_tune_value_lr": FLAGS.fine_tune_value_lr,
    "entropy_decay_duration": FLAGS.entropy_decay_duration, 
    "transfer_policy_minimum_entropy": FLAGS.transfer_policy_minimum_entropy,
    "transfer_policy": FLAGS.transfer_policy,
    "consensus_imitation": FLAGS.consensus_imitation,
    "regret_calculation_steps": FLAGS.regret_calculation_steps,
    "sims_per_entry": FLAGS.sims_per_entry,
    "recovery_window": FLAGS.recovery_window,
    "pretrained_policy_steps": FLAGS.pretrained_policy_steps,
    "save_models": FLAGS.save_models,
    "save_model_path": FLAGS.save_model_path, 
    "load_models": FLAGS.load_models,
    "num_iterations_load_only": FLAGS.num_iterations_load_only,
    "sac_value_clip": FLAGS.value_clip,
    "sac_alpha": FLAGS.alpha, 
    "sac_batch_size": FLAGS.sac_batch_size,
    "sac_update_every": FLAGS.sac_update_every,
    "exploration_policy_type": FLAGS.exploration_policy_type
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
      prd_gamma=0,  # TODO: CHANGED THIS
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

    # TODO: Insert conditional that if we are in regret mode, ensure we are 1. loading models 2. num_load_model_iterations > 0 and 3. save_folder_path is not None and 4. gpsro_iteration < num_load_iterations
    # TODO: Insert a conditional here that, if we are in regret mode, we can call g_psro_solver.set_meta_game and g_psro_solver.set_meta_strategies
    if FLAGS.regret_calculation_mode:
      assert FLAGS.load_models and FLAGS.num_iterations_load_only > 0 and FLAGS.save_folder_path != None 
      if gpsro_iteration >= FLAGS.num_iterations_load_only: # NOTE: this conditional might be wrong
        # Read from the respective iteration data in save_folder_path
        all_files = [f for f in os.listdir(FLAGS.save_folder_path) if os.path.isfile(os.path.join(FLAGS.save_folder_path, f))]
        save_data_path = [file for file in all_files if "iteration_{}.".format(gpsro_iteration-FLAGS.num_iterations_load_only) in file][0] # NOTE: This might be wrong
        save_data_path = FLAGS.save_folder_path + "/" + save_data_path
        print("Retrieving data from save_data_path: ", save_data_path)
        with open(save_data_path, "rb") as npy_file:
          array_list = np.load(npy_file, allow_pickle=True)
        meta_probabilities, _, _, _, _, _ = array_list

        if gpsro_iteration == FLAGS.num_iterations_load_only:
          all_files = [f for f in os.listdir(FLAGS.save_folder_path) if os.path.isfile(os.path.join(FLAGS.save_folder_path, f))]
          save_data_path = [file for file in all_files if "iteration_{}.".format(gpsro_iteration-1) in file][0] # NOTE: This might be wrong
          save_data_path = FLAGS.save_folder_path + "/" + save_data_path
          with open(save_data_path, "rb") as npy_file:
            array_list = np.load(npy_file, allow_pickle=True)
          _, utilities, _, _, _, _ = array_list
          print("Loading the meta game from save_data_path: ", save_data_path)
          g_psro_solver.set_meta_game(utilities)

        
        curr_meta_strategy = g_psro_solver.get_meta_strategies()
        new_meta_strategy = []

        for i, strategy in enumerate(curr_meta_strategy):
          curr = np.zeros(strategy.size)
          for j, elem in enumerate(meta_probabilities[i]):
            curr[j] = elem
          new_meta_strategy.append(curr)
        print("Using saved meta strategy: ", new_meta_strategy)
        g_psro_solver.set_meta_strategies(new_meta_strategy)
        print("Confirming new meta strategy: ", g_psro_solver.get_meta_strategies())
        
    

    if gpsro_iteration < 30 and FLAGS.regret_calculation_mode:
      g_psro_solver._sims_per_entry = 1
    else:
      g_psro_solver._sims_per_entry = FLAGS.sims_per_entry

    print("Sims per entry set to: ", g_psro_solver._sims_per_entry)
    g_psro_solver.iteration()
    
    training_returns = oracle.get_training_returns()
    regret_training_returns = oracle.get_training_regret_returns()
    pure_br_returns = oracle.get_pure_br_returns()
    meta_game = g_psro_solver.get_meta_game()
    meta_probabilities = g_psro_solver.get_meta_strategies()
    policies = g_psro_solver.get_policies()

    g_psro_solver.update_regret_threshold(FLAGS.regret_lambda_final, FLAGS.regret_steps - gpsro_iteration - 1)
    g_psro_solver.update_explore_threshold(FLAGS.final_exploration, FLAGS.regret_steps - gpsro_iteration - 1)

    """
    # NOTE: The following was for regret confirmation of max welfare profiles
    # TODO: Manually set the meta_strategy here using RRD sims!!!! 

    def _rrd_sims(meta_games):
      from open_spiel.python.algorithms import projected_replicator_dynamics
      NUM_ITER = 100
      max_welfare = 0
      max_welfare_profile = []
      print("for ", NUM_ITER, "")
      import random
      for _ in range(NUM_ITER):
        random_nums = [np.array([random.randint(1,5) for _ in range(len(meta_games[0]))]) for _ in range(2)]
        random_profile = [player_profile / np.sum(player_profile) for player_profile in random_nums]
        #prd_dt default = 1e-3 (0.001)
        prd_profile = projected_replicator_dynamics.regularized_replicator_dynamics(
          meta_games,regret_lambda=0.001,
          prd_initial_strategies=random_profile, prd_dt=1e-3, symmetric=True, prd_iterations=int(1e5))
      
        combined_profile = []
        for prob in prd_profile[0]:
          for p2_prob in prd_profile[1]:
            combined_profile.append(prob * p2_prob)
        welfare = np.mean([np.dot(combined_profile, np.array(meta_games[0]).flatten()), np.dot(combined_profile, np.array(meta_games[1]).flatten())])
        if welfare > max_welfare:
          max_welfare_profile = prd_profile
          max_welfare = welfare
      return max_welfare_profile, max_welfare 


    if FLAGS.regret_calculation_mode and gpsro_iteration == 29:
        all_files = [f for f in os.listdir(FLAGS.save_folder_path) if os.path.isfile(os.path.join(FLAGS.save_folder_path, f))]
        save_data_path = [file for file in all_files if "iteration_{}.".format(29) in file][0] # NOTE: This might be wrong
        save_data_path = FLAGS.save_folder_path + "/" + save_data_path
        with open(save_data_path, "rb") as npy_file:
          array_list = np.load(npy_file, allow_pickle=True)
        _, utilities, _, _, _, _ = array_list
        print("Loading the meta game from save_data_path: ", save_data_path)
        g_psro_solver.set_meta_game(utilities)
        meta_game = g_psro_solver.get_meta_game()
        
    if gpsro_iteration == 29:
      print("Starting RRD sims...")
      profile, max_welfare = _rrd_sims(g_psro_solver.get_meta_game())
      print("Found profile with max welfare of : ", 2*max_welfare, " ... setting meta strategy to this: ", profile)
      
      g_psro_solver.set_meta_strategies(profile)
      meta_probabilities = g_psro_solver.get_meta_strategies()

    """
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
      if gpsro_iteration >= FLAGS.num_iterations_load_only: 
        save_iteration_data(gpsro_iteration, meta_probabilities, meta_game, save_folder_path, training_returns, regret_training_returns, config.ppo_training_data, pure_br_returns)

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
    obs_type = rl_environment.ObservationType.OBSERVATION
  elif FLAGS.game_name=="bargaining":
    game = pyspiel.load_game(FLAGS.game_name, {"discount": 0.95})
    obs_type = rl_environment.ObservationType.INFORMATION_STATE
  else:
    game = pyspiel.load_game(FLAGS.game_name)  # The iterated prisoners dilemma does not have "players" info type
    obs_type = rl_environment.ObservationType.INFORMATION_STATE

  env = rl_environment.Environment(game, observation_type=obs_type)

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
  tf.reset_default_graph()

if __name__ == "__main__":
  app.run(main)
