"""
This file is dedicated solely to testing and analyzing offline policy evaluation. 
"""


import time
import collections
import numpy as np
import os
import pyspiel
import tensorflow.compat.v1 as tf
import sys

from datetime import datetime
from absl import app
from absl import flags
from absl import logging
from open_spiel.python import rl_environment
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.uniform_random_policy import UniformRandomPolicy
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.discrete_perturbed_uniform_random_policy import DiscretePerturbedUniformRandomPolicy
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_evaluation.policy_evaluation import OfflineNormalFormGame
from open_spiel.python.algorithms.offline_psro.utils.utils import Transition

FLAGS = flags.FLAGS

# Dataset choice 
flags.DEFINE_string("dataset_path", "", "Path to file to load the offline dataset")

# Game-related
flags.DEFINE_string("game_name", "kuhn_poker", "Game name.")
flags.DEFINE_integer("num_players", 2, "Number of players in the game")
flags.DEFINE_bool("is_symmetric_game", True, "Flag determining whether the target game is symmetric")

# Evaluation metrics to find 
# Choices for experiments:
    # IS_Estimate_CLT 
    # Covariance_True_CLT
    # Covariance_Estimate_CLT
    # Variance_Estimate_CLT
    # Bayesian_MLE_Correction
    # Covariance_Variance_Ratio
flags.DEFINE_string("experiment_name", "IS_Estimate_CLT", "Specifying what kind of experiment to run for the game_model.")


# Distribution analysis (not for game-solving)
flags.DEFINE_integer("total_trajectories", 100000, "Total number of trajectories to take from our dataset.")
flags.DEFINE_integer("sample_size", 1000, "The number of trajectories to average over for IS policy evaluation. This is ONLY used when we are doing distribution analysis.")
flags.DEFINE_integer("num_samples", 1000, "The number of samples to use for bootstrap resampling. The higher the better but can be computationally intense.")
flags.DEFINE_string("save_graph_path", "../../graphs/IS_estimates.png", "Path to save histogram for distribution analysis of IS estimates.")

flags.DEFINE_float("alpha", .8, "How much to determinism we want to equip the evaluation policies with. ")
flags.DEFINE_float("perturb_alpha", 0, "How much to perturb alpha by for experimental purposes (not really for game-solving)")


# Plot-Related


# Misc
flags.DEFINE_integer("seed", 1, "Seed for random")



def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print("\n\n")
    np.random.seed(FLAGS.seed)

    with open(FLAGS.dataset_path, "rb") as npy_file:
        data = np.load(npy_file, allow_pickle=True)

    data = data[:FLAGS.total_trajectories]
    game_model = OfflineNormalFormGame(data, FLAGS.num_players, FLAGS.is_symmetric_game)

    # Use pyspiel to load the game_name
    pyspiel_game = pyspiel.load_game(FLAGS.game_name)
    logging.info("Loaded game: %s", FLAGS.game_name)

    # Use rl_environment as a wrapper and specify the observation type
    env = rl_environment.Environment(pyspiel_game, observation_type=rl_environment.ObservationType.OBSERVATION)

    new_policy = UniformRandomPolicy(env.action_spec()["num_actions"], env.observation_spec()["info_state"])

    ############### IS_Estimate_CLT Experiment ####################
    # Plot the bootstrap sampled importance sampling estimates for varying sample size (number of trajectories per estimate), number of samples (number of estimates to plot)
    # As of now, the behavior policy is known before training (and data collection), and the evaluation policies are alpha-deterministic policies (with varying alpha).
    if FLAGS.experiment_name == "IS_Estimate_CLT":
        evaluation_policies = []
        for i in range(FLAGS.num_players):
            new_policy = DiscretePerturbedUniformRandomPolicy(env.action_spec()["num_actions"], env.observation_spec()["info_state"], alpha=FLAGS.alpha, seed=FLAGS.seed)
            evaluation_policies.append(new_policy)
            new_policy.train(data, players=[i])
        game_model.gaussian_plots_for_is_estimates(behavior_policies, evaluation_policies, FLAGS.sample_size, FLAGS.num_samples, FLAGS.save_graph_path, player=0)
    elif FLAGS.experiment_name == "Covariance_True_CLT":
        # Get the behavior policies as above
        behavior_policies = [new_policy] * FLAGS.num_players 
        evaluation_policies_1 = []
        for i in range(FLAGS.num_players):
            new_policy = DiscretePerturbedUniformRandomPolicy(env.action_spec()["num_actions"], env.observation_spec()["info_state"], alpha=FLAGS.alpha, seed=FLAGS.seed)
            evaluation_policies_1.append(new_policy)
            new_policy.train(data, players=[i])
        evaluation_policies_2 = [evaluation_policies_1[0].create_copy_with_noise(noise=FLAGS.perturb_alpha), evaluation_policies_1[1]]
        game_model.visualize_covariance_between_is_estimates(behavior_policies, evaluation_policies_1, evaluation_policies_2, FLAGS.sample_size, FLAGS.num_samples, FLAGS.save_graph_path, player=0)
        # Measure the covariance between the IS-sampled estimates and plot on 2-d plane. Varying levels of perturbation would be interesting to see the changing relationship.
    elif FLAGS.experiment_name == "Covariance_Estimate_CLT":
        behavior_policies = [new_policy] * FLAGS.num_players 
        evaluation_policies_1 = []
        for i in range(FLAGS.num_players):
            new_policy = DiscretePerturbedUniformRandomPolicy(env.action_spec()["num_actions"], env.observation_spec()["info_state"], alpha=FLAGS.alpha, seed=FLAGS.seed)
            evaluation_policies_1.append(new_policy)
            new_policy.train(data, players=[i])
        evaluation_policies_2 = [evaluation_policies_1[0].create_copy_with_noise(noise=FLAGS.perturb_alpha), evaluation_policies_1[1]]
        game_model.compare_bootstrapped_covariance_with_true_covariance(behavior_policies, evaluation_policies_1, evaluation_policies_2, FLAGS.sample_size, FLAGS.num_samples, FLAGS.save_graph_path, player=0, num_datapoints=100)
    elif FLAGS.experiment_name == "Bayesian_MLE_Correction":
        behavior_policies = [new_policy] * FLAGS.num_players 
        evaluation_policies_1 = []
        for i in range(FLAGS.num_players):
            new_policy = DiscretePerturbedUniformRandomPolicy(env.action_spec()["num_actions"], env.observation_spec()["info_state"], alpha=FLAGS.alpha, seed=FLAGS.seed)
            evaluation_policies_1.append(new_policy)
            new_policy.train(data, players=[i])
        evaluation_policies_2 = [evaluation_policies_1[0].create_copy_with_noise(noise=FLAGS.perturb_alpha), evaluation_policies_1[1]]
        game_model.analyze_bayesian_correction(behavior_policies, evaluation_policies_1, evaluation_policies_2, FLAGS.sample_size, FLAGS.num_samples, FLAGS.save_graph_path, player=0, num_datapoints=100)
    elif FLAGS.experiment_name == "Covariance_Variance_Ratio":
        behavior_policies = [new_policy] * FLAGS.num_players 
        evaluation_policies_1 = []
        for i in range(FLAGS.num_players):
            new_policy = DiscretePerturbedUniformRandomPolicy(env.action_spec()["num_actions"], env.observation_spec()["info_state"], alpha=FLAGS.alpha, seed=FLAGS.seed)
            evaluation_policies_1.append(new_policy)
            new_policy.train(data, players=[i])
        evaluation_policies_2 = [evaluation_policies_1[0].create_copy_with_noise(noise=FLAGS.perturb_alpha), evaluation_policies_1[1]]
        game_model.analyze_covariance_variance_ratio(behavior_policies, evaluation_policies_1, evaluation_policies_2, FLAGS.sample_size, FLAGS.save_graph_path, player=0, num_datapoints=2000)
    else:
        logging.error("Invalid or unimplemented experiment name. ")
        raise NotImplementedError


    # For debugging purposes. Usually, source and target policies will be different
    # game_model.calculate_importance_sampled_evaluation(behavior_policies, evaluation_policies)
    # game_model.get_expected_distribution_difference(evaluation_policies, {}, evaluation_policies, behavior_policies)
    # game_model.map_trajectories_to_strings()
    # game_model.get_true_policy_trajectory_coverage(evaluation_policies, env)

    print("Finished loading of the game model.")
    


if __name__ == "__main__":
  app.run(main)