"""
This file is dedicated solely to testing and analyzing offline target resampling
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
from open_spiel.python.algorithms.offline_psro.B_training.main_components.target_extraction.target_extraction import TargetExtractor
from open_spiel.python.algorithms.offline_psro.utils.utils import generate_single_rollout, compute_hash_string, inverse_hash_string

FLAGS = flags.FLAGS

# Dataset choice 
flags.DEFINE_string("training_dataset_path", "", "Path to file to load the offline dataset")
flags.DEFINE_string('training_initial_steps_dataset_path', "", "Path to file to load the offline dataset of initial steps")

# Game-related
flags.DEFINE_string("game_name", "leduc_poker", "Game name.")
flags.DEFINE_integer("num_players", 2, "Number of players in the game")
flags.DEFINE_bool("is_symmetric_game", True, "Flag determining whether the target game is symmetric")

# Evaluation metrics to find 
# Choices for experiments:
    # ratio_model_with_alpha_policies
flags.DEFINE_string("experiment_name", "ratio_model_with_alpha_policies", "Specifying what kind of experiment to run for the game_model.")

# Dataset and Evaluation Related Parameters
flags.DEFINE_integer("total_trajectories", 100, "Total number of trajectories to take from our dataset.")
flags.DEFINE_string("save_graph_path", "../../graphs/training_loss.png", "Path to save histogram for distribution analysis of ratio training.")

flags.DEFINE_float("alpha", .8, "How much to determinism we want to equip the evaluation policies with. ")
flags.DEFINE_float("perturb_alpha", 0, "How much to perturb alpha by for experimental purposes (not really for game-solving)")

# Ratio Model Related Parameters 
flags.DEFINE_float("ratio_model_discount", .9, "Discount factor for ratio learning COP-TD")
flags.DEFINE_float("ratio_model_normalization_strength", 0, "Soft ratio normalization strength for COP-TD")
flags.DEFINE_integer("ratio_model_batch_size", 100, "Batch size for learning the ratio model")
flags.DEFINE_float("ratio_model_lr", 3e-4, "Learning rate for deep ratio model")
flags.DEFINE_integer("ratio_model_width", 50, "Width of network for deep ratio model")
flags.DEFINE_integer("ratio_model_num_layers", 2, "Number of hidden layers for ratio model")
flags.DEFINE_integer("ratio_update_target_every", 1, "How often to do a hard update on target ratio network")
flags.DEFINE_integer("number_of_training_steps", 10000, "How many gradient steps?")

# Plot-Related

# Misc
flags.DEFINE_integer("seed", 1, "Seed for random")

def run_rollout_for_estimate_of_state_occupancy(number_of_episodes, relevant_players, env, policies, allowed_global_states):
    trajectory_data = []
    for _ in range(number_of_episodes):
        _, trajectory_for_policy_training, _ = generate_single_rollout(FLAGS.num_players, env, policies, env.is_turn_based)
        trajectory_data.extend(trajectory_for_policy_training[relevant_players[0]])
    
    # Count the state visitations 
    global_state_to_count = {}
    for t in trajectory_data: 
        hash_string = compute_hash_string(t.global_state)
        global_state_to_count[hash_string] = global_state_to_count.get(hash_string, 0) + 1

    # Estimate the state occupancy through normalization 
    return {s: float(count) / len(trajectory_data) for s, count in global_state_to_count.items()}




def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print("\n\n")
    np.random.seed(FLAGS.seed)

    with open(FLAGS.training_dataset_path, "rb") as npy_file:
        training_data = np.load(npy_file, allow_pickle=True)

    with open(FLAGS.training_initial_steps_dataset_path, "rb") as npy_file:
        training_initial_steps_data = np.load(npy_file, allow_pickle=True)

    # Combine the datasets
    # Each element in training_data/training_initial_steps_data is a player's data 
    # Each element in a player's data is a trajectory, whether it be list of Timesteps for an entire trajectory or a one-element list for the first step of the trajectory 
    data = []
    for p in range(FLAGS.num_players):
        curr_player_data = []
        for trajectory, initial_steps in zip(training_data, training_initial_steps_data):
            curr_player_trajectory = trajectory[p]
            curr_player_initial_step = initial_steps[p]
            
            curr_player_data.append(curr_player_initial_step + curr_player_trajectory)

            assert curr_player_initial_step[0].is_first == 1

        print("Length of current player dataset: ", len(curr_player_data))
                
        data.append(curr_player_data)

    # Use pyspiel to load the game_name
    pyspiel_game = pyspiel.load_game(FLAGS.game_name)
    logging.info("Loaded game: %s", FLAGS.game_name)

    # NOTE: This is a hyperparameter technically
    relevant_players = [0]

    # Use rl_environment as a wrapper and specify the observation type
    env = rl_environment.Environment(pyspiel_game, observation_type=rl_environment.ObservationType.INFORMATION_STATE)

    # Target Extractor Parameters
    target_extractor_parameters = {
        "num_players": FLAGS.num_players,
        "num_actions": env.action_spec()["num_actions"],
        "relevant_players": relevant_players,
        "ratio_model_discount": FLAGS.ratio_model_discount,
        "ratio_model_normalization_strength": FLAGS.ratio_model_normalization_strength,
        "ratio_model_batch_size": FLAGS.ratio_model_batch_size, 
        "ratio_model_lr": FLAGS.ratio_model_lr, 
        "ratio_model_width": FLAGS.ratio_model_width,
        "ratio_model_num_layers": FLAGS.ratio_model_num_layers,
        "info_state_size": env.observation_spec()["info_state"][0],
        "ratio_model_state_size": env.observation_spec()["info_state"][0] * FLAGS.num_players, 
        "ratio_update_target_every": FLAGS.ratio_update_target_every
    }


    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)

    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        uniform_policy = UniformRandomPolicy(env.action_spec()["num_actions"], env.observation_spec()["info_state"])
        
        ############### ratio_model_with_alpha_policies ####################
        # Train a ratio model that attempts to learn marginalized state-occupancy ratios for policies with limited determinism.
        if FLAGS.experiment_name == "ratio_model_with_alpha_policies":
            behavior_policies = [uniform_policy] * FLAGS.num_players 
            evaluation_policies = []
            for i in range(FLAGS.num_players):
                new_policy = DiscretePerturbedUniformRandomPolicy(env.action_spec()["num_actions"], env.observation_spec()["info_state"], alpha=FLAGS.alpha, seed=FLAGS.seed)
                evaluation_policies.append(new_policy)
                new_policy.train(data[i][:FLAGS.total_trajectories])
            target_extractor = TargetExtractor(sess, data, behavior_policies, evaluation_policies, target_extractor_parameters)

            # IS estimate
            is_global_state_to_ratio_list = {} 
            for traj in data[relevant_players[0]][:FLAGS.total_trajectories]:
                curr_ratio = 1

                # Should be:
                    # First transition: get [next] global state (first step has same curr+next global states) and probability ratio uses the betweens of the first states
                    # Second transition: get the NEXT global state (because the previous global state is the same as first transition). This should be equal to previous times all betweens and actions 
                    # Repeat until the very end. However, if it is DONE, then we don't even care about the next state it is in. Do NOT assign a ratio to that state
                for transition in traj:
                    # Multiply in the between players 
                    for s, a, p, mask in zip(transition.between_info_states, transition.between_actions, transition.between_players, transition.between_legal_actions_masks):
                        # print('between')
                        behavior_prob = behavior_policies[p].probabilities_with_actions([s], [[a]], [mask], numpy=True)[0][0]
                        eval_prob = evaluation_policies[p].probabilities_with_actions([s], [[a]], [mask], numpy=True)[0][0]
                        curr_ratio *= eval_prob / behavior_prob 

                    # Get the next global state 
                    next_global_state_hash = compute_hash_string(transition.next_global_state)

                    # If not is first,  multiply in the current action 
                    if not transition.is_first: # we do not want to double count the first timestep's action
                        eval_prob = evaluation_policies[relevant_players[0]].probabilities_with_actions([transition.info_state], [[transition.action]], [transition.legal_actions_mask], numpy=True)[0][0]
                        behavior_prob = behavior_policies[relevant_players[0]].probabilities_with_actions([transition.info_state], [[transition.action]], [transition.legal_actions_mask], numpy=True)[0][0]
                        curr_ratio *= eval_prob / behavior_prob

                    # If not last, add it to the list
                    if not transition.done:
                        is_global_state_to_ratio_list[next_global_state_hash] = is_global_state_to_ratio_list.get(next_global_state_hash, []) + [curr_ratio]


 
            test_global_state_set = is_global_state_to_ratio_list.keys() 

            # Calculating the true ratios given the dataset! 
            num_episodes_for_evaluation = 200000
            behavior_data = []
            start = time.time()
            while len(behavior_data) < num_episodes_for_evaluation:
                _, trajectory_for_policy_training, first_steps_for_policy_training = generate_single_rollout(FLAGS.num_players, env, behavior_policies, env.is_turn_based)

                trajectory_in_question = first_steps_for_policy_training[relevant_players[0]] + trajectory_for_policy_training[relevant_players[0]]
                if all([compute_hash_string(t.global_state) in test_global_state_set for t in trajectory_in_question]):
                    behavior_data.append(trajectory_in_question)

            logging.info("Generated all relevant rollouts in {} seconds.".format(time.time() - start))
            
            # Calculate the product of ratios for each timestep and separate by global state 
            global_state_to_ratio_list = {}
            for traj in behavior_data:
                curr_ratio = 1
                for transition in traj:
                    # Multiply in the between players 
                    # NOTE: Could make this more efficient by batching by player
                    for s, a, p, mask in zip(transition.between_info_states, transition.between_actions, transition.between_players, transition.between_legal_actions_masks):
                        behavior_prob = behavior_policies[p].probabilities_with_actions([s], [[a]], [mask], numpy=True)[0][0]
                        eval_prob = evaluation_policies[p].probabilities_with_actions([s], [[a]], [mask], numpy=True)[0][0]
                        curr_ratio *= eval_prob / behavior_prob 


                    # Get the next global state 
                    next_global_state_hash = compute_hash_string(transition.next_global_state)


                    # Multiply in the current action 
                    if not transition.is_first: # we do not want to double count the first timestep's action
                        eval_prob = evaluation_policies[relevant_players[0]].probabilities_with_actions([transition.info_state], [[transition.action]], [transition.legal_actions_mask], numpy=True)[0][0]
                        behavior_prob = behavior_policies[relevant_players[0]].probabilities_with_actions([transition.info_state], [[transition.action]], [transition.legal_actions_mask], numpy=True)[0][0]
                        curr_ratio *= eval_prob / behavior_prob

                    # Store the curr_ratio and correspond with global_state 
                    if not transition.done:
                        global_state_to_ratio_list[next_global_state_hash] = global_state_to_ratio_list.get(next_global_state_hash, []) + [curr_ratio]

            # Calculate the average across all global states 

            global_state_to_true_ratio = {k: np.mean(v) for k, v in global_state_to_ratio_list.items()}
            logging.info("Estimated true ratios. ")


            is_global_state_to_ratio = {k: np.mean(v) for k, v in is_global_state_to_ratio_list.items()}
            total_datapoints = float(sum([len(v) for v in is_global_state_to_ratio_list.values()]))
            global_state_to_loss_weight = {k: len(v) / total_datapoints for k, v in is_global_state_to_ratio_list.items()}

            true_normalizer = np.sum(list(global_state_to_true_ratio.values()))
            is_normalizer = np.sum(list(is_global_state_to_ratio.values()))
 
            is_hellinger = 0
            for hash_s in test_global_state_set:
                true_prob = global_state_to_true_ratio[hash_s] / true_normalizer
                is_prob = is_global_state_to_ratio[hash_s] / is_normalizer
                print("IS term: ", ((true_prob) ** (.5) - (is_prob) ** (.5)) ** 2, true_prob, is_prob, is_global_state_to_ratio_list[hash_s])
                is_hellinger += ((true_prob) ** (.5) - (is_prob) ** (.5)) ** 2

            is_hellinger = (is_hellinger ** .5) / (2 ** .5) 

            # Evaluate! Calculate the total variation distance of the model with respect to true distribution.
            global_state_to_ratio_model_values = {}
            for hash_s in test_global_state_set:
                global_state_to_ratio_model_values[hash_s] = target_extractor.predict_ratio(np.array(inverse_hash_string(hash_s), dtype='f').reshape([1, -1]), numpy=True)

            model_normalizer = np.sum(list(global_state_to_ratio_model_values.values()))

            no_training_hellinger = 0
            for hash_s in test_global_state_set:
                true_prob = global_state_to_true_ratio[hash_s] / true_normalizer
                model_prob = global_state_to_ratio_model_values[hash_s] / model_normalizer
                
                no_training_hellinger += ((true_prob) ** (.5) - (model_prob) ** (.5)) ** 2

            no_training_hellinger = (no_training_hellinger ** .5) / (2 ** .5) 

            # Update
            target_extractor.train_ratio_model(number_of_training_steps=FLAGS.number_of_training_steps)

             # Evaluate! Calculate the total variation distance of the model with respect to true distribution.
            global_state_to_ratio_model_values = {}
            for hash_s in test_global_state_set:
                global_state_to_ratio_model_values[hash_s] = target_extractor.predict_ratio(np.array(inverse_hash_string(hash_s), dtype='f').reshape([1, -1]), numpy=True)

            model_normalizer = np.sum(list(global_state_to_ratio_model_values.values()))

            with_training_hellinger = 0
            for hash_s in test_global_state_set:
                true_prob = global_state_to_true_ratio[hash_s] / true_normalizer
                model_prob = global_state_to_ratio_model_values[hash_s] / model_normalizer
                with_training_hellinger += ((true_prob) ** (.5) - (model_prob) ** (.5)) ** 2

            with_training_hellinger = (no_training_hellinger ** .5) / (2 ** .5) 

            print("IS: ", is_hellinger)
            print("No training: ", no_training_hellinger)
            print("Model: ", with_training_hellinger)
            # print("Training model TVD: ", model_tvds)
            # window = 50
            # averaged_model_tvds = [sum(model_tvds[i:i+window])/window for i in range(len(model_tvds) - window + 1)]
            # plt.plot(averaged_model_tvds)
            # plt.savefig("tvd_target_resampling.png")
        else:
            logging.error("Invalid or unimplemented experiment name. ")
            raise NotImplementedError


if __name__ == "__main__":
  app.run(main)