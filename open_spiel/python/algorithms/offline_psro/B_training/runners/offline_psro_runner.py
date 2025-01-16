"""
This is the main offline PSRO code runner. It will combine every combine within this folder.
"""

import time
import numpy as np
import os
import matplotlib.pyplot as plt
import collections 
from datetime import datetime
from absl import app
from absl import flags
from absl import logging
import open_spiel.python.games


from contextlib import ExitStack 
# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# pylint: disable=g-bad-import-order
import pyspiel
import tensorflow.compat.v1 as tf
import sys
from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.bargaining_true_state_generator import BargainingTrueStateGenerator
from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.bargaining_generalized_true_state_generator import BargainingGeneralizedTrueStateGenerator
from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.leduc_poker_true_state_generator import LeducPokerTrueStateGenerator
from open_spiel.python.algorithms.offline_psro.A_pre_training.dataset_generator import DatasetGenerator
from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.bargaining_smart_random_policy import BargainingUniformRandomPolicy
from open_spiel.python.algorithms.offline_psro.B_training.main_components.offline_psro_wrapper import OfflineModelBasedPSRO
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.uniform_random_policy import UniformRandomPolicy
from open_spiel.python.algorithms.offline_psro.B_training.tf_model_management.tf_model_management import TFModelManagement
from open_spiel.python import rl_environment
from open_spiel.python.algorithms.offline_psro.utils.utils import get_terminal_state, Normalizer

# pylint: enable=g-bad-import-order
FLAGS = flags.FLAGS 

# Dataset choice 
flags.DEFINE_string("experiment_name", "test_offline_psro", "Name of the experiment we are running")
flags.DEFINE_string("dataset_path", "", "Path to file to load the offline dataset")
flags.DEFINE_string("game_name", "", "Name of the game we are analyzing")
flags.DEFINE_bool("symmetric", False, "Is the game symmetric?")
flags.DEFINE_integer("num_players", 2, "Number of players in the game")
flags.DEFINE_string("save_path", "random_runs/", "Folder to save all models and data")

# Model Training Related 
flags.DEFINE_string("model_type", "deterministic", "Options are deterministic or stochastic")
flags.DEFINE_integer("model_width", 250, "Width of the model")
flags.DEFINE_integer("model_depth", 2, "Number of hidden layers in model")
flags.DEFINE_integer("ensemble_size", 4, "Number of models in the ensemble")
flags.DEFINE_integer("model_batch_size", 64, "Batch size of learning dynamics model")
flags.DEFINE_float("model_learning_rate", 3e-4, "Learning rate for model training") 
flags.DEFINE_integer("model_training_steps", int(1e4), "Number of training steps to train ensemble model")
flags.DEFINE_bool("use_action_index", False, "Whether or not to directly input the action index to model")

# Best-Response Related
flags.DEFINE_float("mopo_penalty", 1, "How much to weight the state dynamics discrepancy in MOPO fashion")

# DQN/DDQN Related
flags.DEFINE_bool("double", True, "Double DQN or normal DQN?")
flags.DEFINE_integer("dqn_model_width", 200, "Width of the model")
flags.DEFINE_integer("dqn_model_depth", 2, "Number of hidden layers in model")
flags.DEFINE_integer("replay_buffer_size", int(5e4), "Maximum capacity of replay buffer")
flags.DEFINE_integer('dqn_batch_size', 64, "Batch size for training")
flags.DEFINE_float("dqn_learning_rate", 1e-4, "Learning rate for policy training")
flags.DEFINE_integer("update_target_every", 1000, "How many steps before update target network")
flags.DEFINE_integer("learn_every", 2, "Make a gradient step every")
flags.DEFINE_float("discount", .99, "Discount factor")
flags.DEFINE_integer("min_buffer_size_learn", int(5e4), "How many transitions should be in buffer before training")
flags.DEFINE_float("epsilon_start", 1.0, "Exploration parameter start")
flags.DEFINE_float("epsilon_end", .02, "Exploration parameter end")
flags.DEFINE_integer("epsilon_decay_duration", int(2e5), "How many steps to anneal epsilon exploration")
flags.DEFINE_integer("num_training_steps", int(2e5), "Number of step sto train the best response for")

# PSRO Related
flags.DEFINE_integer("num_psro_iterations", 30, "Number of PSRO iterations to execute")
flags.DEFINE_integer('num_simulations', 1000, "Number of simulations to estimate normal-form game entries")
flags.DEFINE_string("meta_strategy_solver", "r3d", "String describing the meta-strategy solver used")
flags.DEFINE_float("alpha", 0.0, "Probability of sampling uniformly from all strategies and setting reward weight to 0")
flags.DEFINE_float("steps_anneal_alpha", 10, "Number of steps to take to anneal alpha")

# DatasetGenerator
flags.DEFINE_string("dataset_generator_behavior_policy", "UniformRandom", "If we are generating the dataset now, behavior policy to use")
flags.DEFINE_integer("dataset_generator_num_datapoints", 1000, "Number of datapoints (default trajectories) to generate for current dataset")

# RRD Related
flags.DEFINE_float("rrd_regret_threshold", 5.0, "Regret threshold for RRD")
flags.DEFINE_float("num_steps_anneal", 20, "Number of psro iterations to anneal rrd regret across")

# Misc
flags.DEFINE_integer("seed", 1, "Seed for Numpy and Tensorflow")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    ############### Display Hyperparameters ########################
    flag_dictionary = FLAGS.flag_values_dict()
    print("Hyperparameters: ")
    for variable_name, variable_value in sorted(flag_dictionary.items()):
        print("{}: {}".format(variable_name, variable_value))
    print("\n\n")
    ############# Display Hyperparameters End ######################

    ##################### Set Seeds ################################
    np.random.seed(seed=FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    ################### Set Seeds End ##############################


    ############# Dataset Extraction + Pyspiel Game ################
    if len(FLAGS.dataset_path) > 0:
        logging.info("Using pre-generated dataset at path {}. Loading Now.".format(FLAGS.dataset_path))
        with open(FLAGS.dataset_path, "rb") as npy_file:
            trajectories = np.load(npy_file, allow_pickle=True)
    else: 
        logging.info("Generating new dataset using behavior policies: ", )
        behavior_policies = [FLAGS.dataset_generator_behavior_policy] * FLAGS.num_players
        generator = DatasetGenerator(behavior_policies, FLAGS.num_players, FLAGS.game_name, "", FLAGS.symmetric, FLAGS.dataset_generator_num_datapoints)

        new_trajectories = generator.generate_rollouts(save_data=False)

        trajectories = np.empty(len(new_trajectories), object)
        trajectories[:] = new_trajectories
    
    data = []
    for rollout in trajectories:
        data.extend(rollout)

    pyspiel_game = pyspiel.load_game(FLAGS.game_name)
    logging.info("Loaded game: %s", FLAGS.game_name)
    ############ Dataset Extraction + Pyspiel Game End #############

    ################ Game-Specific Modules #########################
    if FLAGS.game_name == "bargaining":
        true_state_extractor = BargainingTrueStateGenerator(FLAGS.game_name, pyspiel_game.information_state_tensor_shape()[0])
        max_episode_length = pyspiel_game.max_game_length()
        reward_structure = "terminal"
    if FLAGS.game_name == "bargaining_generalized":
        true_state_extractor = BargainingGeneralizedTrueStateGenerator(FLAGS.game_name, pyspiel_game.information_state_tensor_shape()[0])
        max_episode_length = pyspiel_game.max_game_length()
        reward_structure = "terminal"
    if FLAGS.game_name == "leduc_poker":
        true_state_extractor = LeducPokerTrueStateGenerator(FLAGS.game_name)
        reward_structure = "terminal"
    true_state_extractor.get_set_info_depending_on_game(pyspiel_game)
    ############## Game-Specific Modules End #######################

    ################ Globals #######################
    env = rl_environment.Environment(pyspiel_game, observation_type=rl_environment.ObservationType.INFORMATION_STATE)
    tf_model_management_module = TFModelManagement()
    state_size = len(data[0].global_state)
    action_size = (1 if env.is_turn_based else env.num_players) if FLAGS.use_action_index else (len(true_state_extractor.action_index_to_vector_representation(0)) if env.is_turn_based else len(true_state_extractor.action_index_to_offer_representation(0)) * 2)
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)
    assert FLAGS.epsilon_decay_duration <= FLAGS.num_training_steps
    ############## Globals End #####################

    if FLAGS.experiment_name == "test_offline_psro":
        ############################## PSRO Initialization ###############################
        if FLAGS.game_name == "bargaining":
            initial_strategies = [UniformRandomPolicy(num_actions=env.action_spec()["num_actions"], state_size=1) for _ in range(1 if FLAGS.symmetric else FLAGS.num_players)]
            #initial_strategies = [UniformRandomPolicy(true_state_extractor, minimum_acceptance_probability=.1, pyspiel_game=pyspiel_game, num_actions=env.action_spec()["num_actions"], state_size=env.observation_spec()["info_state"]) for _ in range(1 if FLAGS.symmetric else FLAGS.num_players)]
        else:
            initial_strategies = [UniformRandomPolicy(num_actions=env.action_spec()["num_actions"], state_size=1) for _ in range(1 if FLAGS.symmetric else FLAGS.num_players)]

        reward_normalizer = Normalizer(mean=0, standard_deviation=np.std([d.rewards for d in data]))
        state_normalizer = Normalizer(mean=np.mean([d.global_state for d in data]), standard_deviation=np.std([d.global_state for d in data]))
        action_normalizer = Normalizer(mean=np.mean([[true_state_extractor.action_index_to_vector_representation(a) for a in d.actions if a != None] for d in data]), standard_deviation=np.std([[true_state_extractor.action_index_to_vector_representation(a) for a in d.actions if a != None] for d in data]))
        observation_normalizer = Normalizer(mean=np.mean([d.info_states[d.relevant_players[0]] for d in data]), standard_deviation=np.std([d.info_states[d.relevant_players[0]] for d in data]))
        
        print("Normalizers: ")
        print("Reward: {} \n State: {} \n Action: {} \n Observation: {} \n".format(reward_normalizer, state_normalizer, action_normalizer, observation_normalizer))
        if FLAGS.use_action_index:
            print("Using action index. Ensure that the action normalizer is 0 with standard deviation 1: ", action_normalizer)
        
        # Dynamics model parameters
        model_args = {"model_type":FLAGS.model_type, "state_size":state_size, "action_size":action_size, "true_state_extractor": true_state_extractor,
                        "hidden_sizes": [FLAGS.model_width] * FLAGS.model_depth, "batch_size": FLAGS.model_batch_size, 
                        "learning_rate": FLAGS.model_learning_rate, "ensemble_size": FLAGS.ensemble_size,
                        "observation_sizes":[env._game.observation_tensor_shape()[0]] * env.num_players,
                        "num_players":env.num_players, "num_actions":env.action_spec()["num_actions"], "turn_based": env.is_turn_based,
                        "training_steps": FLAGS.model_training_steps, "indicator_rounding": False, "reward_structure": reward_structure, 
                        "data": data, "use_action_index": FLAGS.use_action_index,
                        "state_normalizer": state_normalizer, "action_normalizer": action_normalizer, "reward_normalizer": reward_normalizer, 
                        "observation_normalizer": observation_normalizer}

        # Initialization of Offline PSRO Looper
        with tf.Session(config=session_conf) as sess:
            sess.run(tf.global_variables_initializer())
            psroLooper = OfflineModelBasedPSRO(trajectories, FLAGS.symmetric, env.is_turn_based, FLAGS.num_players, env.action_spec()["num_actions"], initial_strategies, 
                                            model_args, tf_model_management_module, FLAGS.save_path, true_state_extractor, max_episode_length,
                                            verbose=True)

        # Response training parameters
        response_parameters = {
                "state_representation_size": env._game.information_state_tensor_shape()[0], "num_actions": env.action_spec()["num_actions"],
                "double": FLAGS.double, "hidden_layers_sizes": [FLAGS.dqn_model_width] * FLAGS.dqn_model_depth, 
                "replay_buffer_capacity": FLAGS.replay_buffer_size, "batch_size": FLAGS.dqn_batch_size, 
                "learning_rate": FLAGS.dqn_learning_rate, "update_target_network_every": FLAGS.update_target_every, 
                "learn_every": FLAGS.learn_every, "discount_factor": FLAGS.discount, "min_buffer_size_to_learn": FLAGS.min_buffer_size_learn, 
                "epsilon_start": FLAGS.epsilon_start, "epsilon_end": FLAGS.epsilon_end, "epsilon_decay_duration": FLAGS.epsilon_decay_duration,
            }


        # Initialize the empirical game
        start = time.time()
        psroLooper.update_empirical_game(FLAGS.num_simulations)
        print("Updating empirical game for {} simulations took {} seconds. ".format(FLAGS.num_simulations, time.time() - start))

        # Initialize profiles (needed if an initial strategy was created instead of inputted)
        psroLooper.update_profile(FLAGS.meta_strategy_solver, {"regret_lambda": FLAGS.rrd_regret_threshold})
        curr_regret_lambda = FLAGS.rrd_regret_threshold
        curr_alpha = FLAGS.alpha
        ############################ PSRO Initialization End #############################

        ############################ Main PSRO Loop ######################################
        for iteration in range(FLAGS.num_psro_iterations):
            ###################### Response Training ########################
            players_need_response_trained = list(range(FLAGS.num_players)) if not FLAGS.symmetric else [0]
            for p in players_need_response_trained:
                start = time.time()
                psroLooper.train_and_add_strategy(response_parameters, training_player=p, num_training_steps=FLAGS.num_training_steps, alpha=curr_alpha, mopo_penalty=FLAGS.mopo_penalty)
                print("Training player {}'s strategy took {} seconds.".format(p, time.time() - start))
            #################### Response Training End ######################

            ############## Empirical Game and Profile Update ################
            start = time.time()
            psroLooper.update_empirical_game(FLAGS.num_simulations)
            print("Updating empirical game took total of {} seconds.\n".format(time.time() - start))
            print("Updating profile with: ", FLAGS.meta_strategy_solver)
            psroLooper.update_profile(FLAGS.meta_strategy_solver, {"regret_lambda": curr_regret_lambda})
            psroLooper.save_current_iteration_data(iteration)
            curr_regret_lambda = max(1e-5, curr_regret_lambda - FLAGS.rrd_regret_threshold / FLAGS.num_steps_anneal)
            curr_alpha = max(0, curr_alpha - FLAGS.alpha / FLAGS.steps_anneal_alpha)
            ############ Empirical Game and Profile Update End ##############
        
        psroLooper.save_empirical_game()
        ########################## Main PSRO Loop End ####################################
    print("Experiment done.")


if __name__ == "__main__":
  app.run(main)

