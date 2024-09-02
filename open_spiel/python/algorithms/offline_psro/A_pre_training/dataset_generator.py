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
import itertools

from datetime import datetime
from absl import app
from absl import flags
from absl import logging
from open_spiel.python import rl_environment
from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.bargaining_true_state_generator import BargainingTrueStateGenerator
from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.leduc_poker_true_state_generator import LeducPokerTrueStateGenerator
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.uniform_random_policy import UniformRandomPolicy
from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.bargaining_smart_random_policy import BargainingUniformRandomPolicy
from open_spiel.python.algorithms.offline_psro.utils.utils import Transition, StateAction, generate_single_rollout, compute_hash_string
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
        if game_name == "bargaining":
            pyspiel_game = pyspiel.load_game(game_name, {"discount": 0.99})
        else:
            pyspiel_game = pyspiel.load_game(game_name)
        logging.info("Loaded game: %s", game_name)

        # Get the game-specific true state extractor 
        if game_name == "bargaining":
            self.true_state_extractor = BargainingTrueStateGenerator(game_name, pyspiel_game.information_state_tensor_shape()[0])
        if game_name == "leduc_poker":
            self.true_state_extractor = LeducPokerTrueStateGenerator(game_name)
        
        self.true_state_extractor.get_set_info_depending_on_game(pyspiel_game)
        self._pyspiel_game = pyspiel_game

        # Use rl_environment as a wrapper and specify the observation type
        # we do OBSERVATION so that we can train State -> Observation mapping. Info_states are created with a sequence of observations later.
        self._env = rl_environment.Environment(pyspiel_game, observation_type=rl_environment.ObservationType.OBSERVATION)  

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
            elif policy_type == "BargainingUniformRandom":
                new_policy = BargainingUniformRandomPolicy(self.true_state_extractor, minimum_acceptance_probability=.1, pyspiel_game=self._pyspiel_game, 
                                                           num_actions=self._env.action_spec()["num_actions"], state_size=self._env.observation_spec()["info_state"])
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
        data = []
        while len(data) < self.num_datapoints:
            trajectory = generate_single_rollout(self.num_players, self._env, self.behavior_policies, [None] * FLAGS.num_players, self._is_turn_based, self.true_state_extractor)  # This is a list of size num_players
            # print("lengths: ", len(trajectory_for_policy_evaluation))
            if self.is_symmetric_game:
                # For each possible permutation of range(self._num_players)
                for new_player_assignment in itertools.permutations(list(range(self.num_players))):
                    # the i-th element in new_player_assignment refers to player i usurping the role of player new_player_assignment[i]
                    curr_trajectory = []
                    # For each of the Transitions in the trajectory
                    for transition in trajectory:
                        # Create a new trajectory that switches indices and modifies the current player based on the permutation.

                        # TODO: Mistake is that our global state likely has to account for player permutations as well!
                        curr_trajectory.append(Transition(
                            info_states=[transition.info_states[new_player_assignment[i]] for i in range(self.num_players)],
                            actions=[transition.actions[new_player_assignment[i]] for i in range(self.num_players)],
                            legal_actions_masks=[transition.legal_actions_masks[new_player_assignment[i]] for i in range(self.num_players)],
                            rewards=[transition.rewards[new_player_assignment[i]] for i in range(self.num_players)],
                            next_info_states=[transition.next_info_states[new_player_assignment[i]] for i in range(self.num_players)],
                            done=transition.done,
                            relevant_players=sorted([new_player_assignment[player] for player in transition.relevant_players]),
                            global_state=self.true_state_extractor.true_state_symmetric_permute(transition.global_state, permutation=new_player_assignment),
                            next_global_state=self.true_state_extractor.true_state_symmetric_permute(transition.next_global_state, permutation=new_player_assignment)
                        ))

                    data.append(curr_trajectory)
            else:
                data.append(trajectory)

        logging.info("Finished creating dataset")

        if save_data:
            self._save_data(data, "model_training")

        return data


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print("\n\n")
    np.random.seed(FLAGS.seed)
    
    if FLAGS.policy_path:
        raise NotImplementedError
    elif FLAGS.game_name == "bargaining":
        behavior_policies = ["BargainingUniformRandom" for _ in range(FLAGS.num_players)]
    else:
        # Initialize a random policy here
        behavior_policies = ["UniformRandom" for _ in range(FLAGS.num_players)]
        
    
    generator = DatasetGenerator(behavior_policies, FLAGS.num_players, FLAGS.game_name, FLAGS.save_dataset_path, FLAGS.is_symmetric_game, FLAGS.num_datapoints)

    generator.generate_rollouts()


if __name__ == "__main__":
  app.run(main)
