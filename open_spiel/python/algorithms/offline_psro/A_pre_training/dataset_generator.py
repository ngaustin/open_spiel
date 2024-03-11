"""
This file is dedicated to generating datasets from any particular game and saving it to the datasets folder. By default, initialized policies are uniformly random, unless a policy is passed in or specified.
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
from open_spiel.python.algorithms.offline_psro.utils.utils import Transition, StateAction, generate_single_rollout

FLAGS = flags.FLAGS

# Game
flags.DEFINE_string("game_name", "kuhn_poker", "Game name.")
flags.DEFINE_integer("num_players", 2, "Number of players in the game")
flags.DEFINE_bool("is_symmetric_game", True, "Flag determining whether the target game is symmetric")
flags.DEFINE_integer("seed", 1, "Numpy seed")

# Dataset settings
flags.DEFINE_integer("num_datapoints", 100, "Number of transitions or trajectories to save to the dataset. If symmetric, then datapoints from both players will contribute")

# Saving and Loading
flags.DEFINE_string("save_dataset_path", "./datasets/", "Path to directory to save the resultant dataset.")
flags.DEFINE_string("policy_path", None, "Path to behavior policies. If None, then initialize to random uniform")
flags.DEFINE_string("policy_description", "UniformRandom", "Description of the policies used to generate the dataset for saving purposes")



class DatasetGenerator:
    def __init__(self, behavior_policy_type, num_players, game_name, save_dataset_path, is_symmetric_game, num_datapoints):
        """
        behavior_policy_type: list of strings specifying the type of policy for each player
        game_name: string that specifies the game 
        save_dataset_path: relative path to save the dataset to
        is_symmetric_game: boolean indicating whether game_name is symmetric or not (will determine the dataset)
        num_datapoints: the number of "transitions" or "trajectories" to create for the dataset. For a symmetric game, we count datapoints contributed by both players
        """
        self.behavior_policy_type = behavior_policy_type
        self.save_dataset_path = save_dataset_path
        self.is_symmetric_game = is_symmetric_game
        self.num_datapoints = num_datapoints
        self.num_players = num_players
        self.behavior_policies = []

        # Use pyspiel to load the game_name
        pyspiel_game = pyspiel.load_game(game_name)
        logging.info("Loaded game: %s", game_name)

        # Use rl_environment as a wrapper and specify the observation type
        self._env = rl_environment.Environment(pyspiel_game, observation_type=rl_environment.ObservationType.INFORMATION_STATE)

        # Is this a turn-based game?
        self._is_turn_based = self._env.is_turn_based

        # Initialize the behavior policies
        self._initialize_behavior_policies()

        # Use the observation and action spec to initialize your behavior policies possibly OR check that the behavior policies appropriately support the observation/action spaces
        for policy in self.behavior_policies:
            if self._env.observation_spec()["info_state"] != policy.state_size or self._env.action_spec()["num_actions"] != policy.num_actions:
                logging.error('Behavior policies do not align in state or action space with the chosen game')
                logging.error('Observation/Action spaces:    Game: %i %i     Policy: %i %i', self._env.observation_spec()["info_state"], self._env.action_spec()["num_actions"], policy.state_size, policy.num_actions)
                raise Exception ('Invalid policy state/action space sizes. ')


    def _initialize_behavior_policies(self):
        for policy_type in self.behavior_policy_type:
            if policy_type == "UniformRandom":
                new_policy = UniformRandomPolicy(self._env.action_spec()["num_actions"], self._env.observation_spec()["info_state"])
            else:
                raise NotImplementedError
            self.behavior_policies.append(new_policy)

    def _save_data(self, data, additional_information="None"):
        """
        Save the generated transitions or trajectories to the specified path. Depending on whether the game is symmetric or not, save one or multiple files (one if symmetric, multiple if not).
        Indicate so in the generated files.
        """
        pathExists = os.path.exists(FLAGS.save_dataset_path)
        if not pathExists:
            os.makedirs(FLAGS.save_dataset_path)

        symmetric_string = "symmetric" if FLAGS.is_symmetric_game else "nonsymmetric"
        
        file_name = "{}_{}_{}_{}_players_{}_points_{}_info_{}.npy".format(str(datetime.now()).replace(' ', '_'), FLAGS.game_name, FLAGS.policy_description, FLAGS.num_players, FLAGS.num_datapoints, symmetric_string, additional_information)
        save_data_path = FLAGS.save_dataset_path + file_name
        self._numpy_save_array_list(data, save_data_path)

        return
    
    def _numpy_save_array_list(self, data, path):
        object_array_list = np.empty(len(data), object)
        object_array_list[:] = data
        with open(path, "wb") as npy_file:
            np.save(npy_file, object_array_list)
        logging.info("Saved data at file: %s", path)

    def generate_rollouts(self, save_data=True):
        """ 
        Parent call to generate a bunch of rollouts and save the result using save_data

        return: a list of all data. Length can be 1 or self.num_players depending on whether the game is symmetric. Each element in the datasets 
                can correspond to the transitions or trajectories themselves.
        """
        data_for_policy_evaluation, data_for_policy_training = [], []
        while len(data_for_policy_evaluation) < self.num_datapoints:
            trajectory_for_policy_evaluation, trajectory_for_policy_training = generate_single_rollout(self.num_players, self._env, self.behavior_policies, self._is_turn_based)  # This is a list of size num_players
            data_for_policy_evaluation.append(trajectory_for_policy_evaluation)
            data_for_policy_training.append(trajectory_for_policy_training)
        
        logging.info("Finished creating dataset")
        
        if save_data:
            self._save_data(data_for_policy_evaluation, "policy_evaluation")
            self._save_data(data_for_policy_training, "policy_training")
        return data_for_policy_evaluation, data_for_policy_training


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print("\n\n")
    np.random.seed(FLAGS.seed)
    
    if FLAGS.policy_path:
        raise NotImplementedError
    else:
        # Initialize a random policy here
        behavior_policies = ["UniformRandom" for _ in range(FLAGS.num_players)]
        
    
    generator = DatasetGenerator(behavior_policies, FLAGS.num_players, FLAGS.game_name, FLAGS.save_dataset_path, FLAGS.is_symmetric_game, FLAGS.num_datapoints)

    generator.generate_rollouts()


if __name__ == "__main__":
  app.run(main)
