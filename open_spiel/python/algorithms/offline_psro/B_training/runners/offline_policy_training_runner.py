"""
This file is dedicated solely to testing and analyzing offline policy training. 
"""


import time
import collections
import numpy as np
import os
import pyspiel
import tensorflow.compat.v1 as tf
import sys

import matplotlib.pyplot as plt

from datetime import datetime
from absl import app
from absl import flags
from absl import logging
from open_spiel.python import rl_environment
from open_spiel.python.algorithms.offline_psro.B_training.tf_model_management.tf_model_management import TFModelManagement
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.uniform_random_policy import UniformRandomPolicy
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.minimalist_td3_bc_gumbel import TD3_BC_Gumbel
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.latent_td3_bc_gumbel import Latent_TD3_BC_Gumbel
from open_spiel.python.algorithms.offline_psro.utils.utils import generate_single_rollout
from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.bargaining_true_state_generator import BargainingTrueStateGenerator
from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.leduc_poker_true_state_generator import LeducPokerTrueStateGenerator

FLAGS = flags.FLAGS

# Dataset choice 
flags.DEFINE_string("training_dataset_path", "", "Path to file to load the offline dataset")

# Game-related
flags.DEFINE_string("game_name", "leduc_poker", "Game name.")
flags.DEFINE_integer("num_players", 2, "Number of players in the game")
flags.DEFINE_bool("is_symmetric_game", True, "Flag determining whether the target game is symmetric")

# Evaluation metrics to find 
# Choices for experiments:
    # ratio_model_with_alpha_policies
flags.DEFINE_string("experiment_name", "minimalist_td3_bc_gumbel", "Specifying what kind of experiment to run for the game_model.")

# Dataset and Evaluation Related Parameters
flags.DEFINE_integer("total_trajectories", 100, "Total number of trajectories to take from our dataset.")
flags.DEFINE_string("save_graph_path", "../../graphs/training_loss.png", "Path to save histogram for distribution analysis of ratio training.")
flags.DEFINE_integer("evaluate_every", 200, "How often to run environment evaluation steps")
flags.DEFINE_integer("number_trials_per_evaluation", 200, "How many trials to average over per evaluation")

# Policy Training Related 
flags.DEFINE_float("policy_network_lr", 3e-4, "Policy network learning rate")
flags.DEFINE_float("value_network_lr", 3e-4, "Value network learning rate")
flags.DEFINE_integer("policy_network_width", 50, "Policy network width")
flags.DEFINE_integer("policy_network_depth", 1, "Policy network depth")
flags.DEFINE_integer("value_network_width", 50, "Value network width")
flags.DEFINE_integer("value_network_depth", 1, "Value network depth")
flags.DEFINE_float("discount", .99, "Discount factor")
flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_float("temp", .4, "Temperature for gumbel-softmax")
flags.DEFINE_float("alpha", 2.5, "Strength of lambda regularization in TD3-BC")
flags.DEFINE_float("tau", .99, "Soft target update for TD3-BC")
flags.DEFINE_integer("num_gradient_steps", 1000, "number of steps")
flags.DEFINE_integer("num_bc_pretrain_steps", 1000, "number pretraining steps")
flags.DEFINE_integer("policy_update_frequency", 2, "update policy every?")

flags.DEFINE_integer("latent_space_size", 50, "latent space size")
flags.DEFINE_integer("encoder_width", 50, "encoder width")
flags.DEFINE_integer("encoder_depth", 1, "encoder depth")
flags.DEFINE_float("triplet_loss_epsilon", 1e-2, "contrastive loss triplet epsilon")
flags.DEFINE_float("contrastive_loss_weight", 0, "contrastive loss weight")

# Plot-Related


# Misc
flags.DEFINE_integer("seed", 1, "Seed for random")
flags.DEFINE_bool("save_and_freeze", False, "Save and freeze?")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print("\n\n")
    np.random.seed(FLAGS.seed)

    with open(FLAGS.training_dataset_path, "rb") as npy_file:
        training_data = np.load(npy_file, allow_pickle=True)

    # Combine the datasets
    # Each element in training_data/training_initial_steps_data is a player's data 
    # Each element in a player's data is a trajectory, whether it be list of Timesteps for an entire trajectory or a one-element list for the first step of the trajectory 
    data = []

    for p in range(FLAGS.num_players):
        curr_player_data = []
        for trajectory in training_data: 
            curr_player_trajectory = trajectory[p]
            curr_player_data.extend(curr_player_trajectory)
                
        data.append(curr_player_data)

    # Use pyspiel to load the game_name
    pyspiel_game = pyspiel.load_game(FLAGS.game_name)
    logging.info("Loaded game: %s", FLAGS.game_name)

    # Get the game-specific true state extractor 
    if FLAGS.game_name == "bargaining":
        true_state_extractor = BargainingTrueStateGenerator(FLAGS.game_name)
    if FLAGS.game_name == "leduc_poker":
        true_state_extractor = LeducPokerTrueStateGenerator(FLAGS.game_name)
    
    true_state_extractor.get_set_info_depending_on_game(pyspiel_game)

    # Use rl_environment as a wrapper and specify the observation type
    env = rl_environment.Environment(pyspiel_game, observation_type=rl_environment.ObservationType.INFORMATION_STATE)

    # NOTE: This is a hyperparameter technically
    relevant_players = [0]


    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)

    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        
    ############### minimalist_td3_bc_gumbel ####################
    if FLAGS.experiment_name == "minimalist_td3_bc_gumbel":
        # Policy Training parameters
        with tf.Session(config=session_conf) as sess:
            model_manager = TFModelManagement()
            sess.run(tf.global_variables_initializer())
            policy_training_parameters = {
                "data":data[relevant_players[0]],
                "session": sess,
                "policy_network_lr": FLAGS.policy_network_lr,
                "value_network_lr": FLAGS.value_network_lr,
                "policy_network_shape": [FLAGS.policy_network_width] * FLAGS.policy_network_depth,
                "value_network_shape": [FLAGS.value_network_width] * FLAGS.value_network_depth, 
                "discount": FLAGS.discount,
                "batch_size": FLAGS.batch_size,
                "temp": FLAGS.temp,
                "alpha": FLAGS.alpha, 
                "tau": FLAGS.tau,
                "policy_update_frequency": FLAGS.policy_update_frequency,
                "encoder_network_shape": [FLAGS.encoder_width] * FLAGS.encoder_depth, 
                "latent_space_size": FLAGS.latent_space_size, 
                "triplet_loss_epsilon": FLAGS.triplet_loss_epsilon,
                "contrastive_loss_weight": FLAGS.contrastive_loss_weight
            }
            # policy = TD3_BC_Gumbel(num_actions=env.action_spec()["num_actions"], state_size=env.observation_spec()["info_state"][0], policy_args=policy_training_parameters)
            policy = Latent_TD3_BC_Gumbel(num_actions=env.action_spec()["num_actions"], state_size=env.observation_spec()["info_state"][0], policy_args=policy_training_parameters)
            response_target = UniformRandomPolicy(env.action_spec()["num_actions"], env.observation_spec()["info_state"])
            
            
            training_time = 0
            evaluation_time = 0 
            total_steps = 0 
            all_evaluation_returns = []

            # policy.pretrain_bc(FLAGS.num_bc_pretrain_steps)

            while total_steps < FLAGS.num_gradient_steps:
                train_time = time.time() 
                policy.train(FLAGS.evaluate_every)
                training_time += time.time() - train_time

                total_steps += FLAGS.evaluate_every
                evaluation_returns = []
                eval_policies = [response_target]
                eval_policies.insert(relevant_players[0], policy)

                eval_time = time.time() 
                for _ in range(FLAGS.number_trials_per_evaluation):
                    rollout, _, _ = generate_single_rollout(FLAGS.num_players, env, eval_policies, env.is_turn_based, true_state_extractor)
                    ret = sum([t.rewards[relevant_players[0]] for t in rollout])
                    evaluation_returns.append(ret)
                
                evaluation_time += time.time() - eval_time
                all_evaluation_returns.append(sum(evaluation_returns) / FLAGS.number_trials_per_evaluation)
                print("Total steps: ", total_steps)
            
            window = 100 
            averaged_all_evaluation_returns = [sum(all_evaluation_returns[i:i+window]) / window for i in range(len(all_evaluation_returns) - window + 1)]
            plt.plot(averaged_all_evaluation_returns)
            plt.savefig("td3_bc_gumbel_round_1_alpha_{}_steps_{}_temp_{}.jpg".format(FLAGS.alpha, FLAGS.num_gradient_steps, FLAGS.temp))
            print("Returns: ", all_evaluation_returns)
            print("Final evaluation: ", all_evaluation_returns[-1])
            logging.info("Policy training finished. {} seconds spent on training. {} second spent on evaluation. ".format(training_time, evaluation_time))

            if FLAGS.save_and_freeze:
                model_manager.save(sess, "../tf_model_management/models/test.ckpt")
        # Outside of session 
        if FLAGS.save_and_freeze:
            model_manager.freeze_graph("../tf_model_management/models/", [policy.get_policy_output_variable_name()])
            frozen_graph = model_manager.load_frozen_graph('../tf_model_management/models/frozen_model.pb')
    else:
        logging.error("Invalid or unimplemented experiment name. ")
        raise NotImplementedError


if __name__ == "__main__":
  app.run(main)