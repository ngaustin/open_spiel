"""
Runs MoREL using DDQN and trained ensemble model
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
import matplotlib.pyplot as plt
from absl import logging
from contextlib import ExitStack 

from open_spiel.python import rl_environment
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.world_model_deterministic import WorldModelDeterministic
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.ddqn_simple import DQN
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.uniform_random_policy import UniformRandomPolicy
from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.bargaining_true_state_generator import BargainingTrueStateGenerator
from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.leduc_poker_true_state_generator import LeducPokerTrueStateGenerator
from open_spiel.python.algorithms.offline_psro.utils.utils import generate_single_rollout, Step, get_graphs_and_context_managers
from open_spiel.python.algorithms.offline_psro.B_training.tf_model_management.tf_model_management import TFModelManagement

FLAGS = flags.FLAGS

# Dataset choice 
flags.DEFINE_string("experiment_name", "test_morel_training", "Name of the experiment we are running")
flags.DEFINE_string("dataset_path", "", "Path to file to load the offline dataset")
flags.DEFINE_string("game_name", "", "Name of the game we are analyzing")
flags.DEFINE_integer("max_episode_length", 10, "Max number of steps in the game")
flags.DEFINE_bool("symmetric", False, "Symmetric?")

# Model Training Related 
flags.DEFINE_integer("model_width", 250, "Width of the model")
flags.DEFINE_integer("model_depth", 2, "Number of hidden layers in model")
flags.DEFINE_integer("ensemble_size", 4, "Number of models in the ensemble")
flags.DEFINE_integer("batch_size", 64, "Batch size of learning dynamics model")
flags.DEFINE_float("learning_rate", 3e-4, "Learning rate for model training") 
flags.DEFINE_float("halt_threshold", 1, "Maximum prediction difference for state-action pair to be considered OOD")
flags.DEFINE_float("penalty", 2, "Penalty for MoREL OOD State-Action pairs HALT")
flags.DEFINE_bool("indicator_rounding", True, "Whether or not to round state predictions")

# Best Response Related
flags.DEFINE_integer("num_training_steps", int(2e5), "How many steps to train BR")
flags.DEFINE_bool("double", True, "Double DQN or normal DQN?")
flags.DEFINE_integer("dqn_model_width", 200, "Width of the model")
flags.DEFINE_integer("dqn_model_depth", 2, "Number of hidden layers in model")
flags.DEFINE_integer("replay_buffer_size", int(5e4), "Maximum capacity of replay buffer")
flags.DEFINE_integer('dqn_batch_size', 64, "Batch size for training")
flags.DEFINE_float("dqn_learning_rate", 1e-4, "Learning rate for policy training")
flags.DEFINE_integer("update_target_every", 1000, "How many steps before update target network")
flags.DEFINE_integer("learn_every", 2, "Make a gradient step every")
flags.DEFINE_float("discount", .99, "Discount factor")
flags.DEFINE_integer("min_buffer_size_learn", 1000, "How many transitions should be in buffer before training")
flags.DEFINE_float("epsilon_start", .8, "Exploration parameter start")
flags.DEFINE_float("epsilon_end", .02, "Exploration parameter end")
flags.DEFINE_integer("epsilon_decay_duration", int(5e4), "How many steps to anneal epsilon exploration")

# Plot-Related
flags.DEFINE_string("save_path", "", "Place to save our data")
flags.DEFINE_string("load_path", "", "Place to load data from")

# Misc
flags.DEFINE_integer("seed", 1, "Seed for random")



def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print("\n\n")
    np.random.seed(FLAGS.seed)

    with open(FLAGS.dataset_path, "rb") as npy_file:
        trajectories = np.load(npy_file, allow_pickle=True)
    data = []
    for rollout in trajectories:
        data.extend(rollout)

    pyspiel_game = pyspiel.load_game(FLAGS.game_name)
    logging.info("Loaded game: %s", FLAGS.game_name)

    # Use rl_environment as a wrapper and specify the observation type
    env = rl_environment.Environment(pyspiel_game, observation_type=rl_environment.ObservationType.INFORMATION_STATE)
    state_size = len(data[0].global_state)
    action_size = 1 if env.is_turn_based else env.num_players

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)

    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
    
    assert FLAGS.epsilon_decay_duration <= FLAGS.num_training_steps


    # Get the game-specific true state extractor 
    if FLAGS.game_name == "bargaining":
        true_state_extractor = BargainingTrueStateGenerator(FLAGS.game_name, pyspiel_game.information_state_tensor_shape()[0])
        reward_structure = "terminal"
    if FLAGS.game_name == "leduc_poker":
        true_state_extractor = LeducPokerTrueStateGenerator(FLAGS.game_name)
        reward_structure = "terminal"
    true_state_extractor.get_set_info_depending_on_game(pyspiel_game)
  
    
    if FLAGS.experiment_name == "test_morel_training":
        rl_training_graph = tf.Graph()
        with tf.Session(config=session_conf, graph=rl_training_graph) as sess:

            model_args = {"hidden_sizes": [FLAGS.model_width] * FLAGS.model_depth,
                        "batch_size": FLAGS.batch_size,
                        "learning_rate": FLAGS.learning_rate,
                        "ensemble_size": FLAGS.ensemble_size,
                        "observation_sizes":[env._game.observation_tensor_shape()[0]] * env.num_players,
                        "num_players":env.num_players,
                        "num_actions":env.action_spec()["num_actions"],
                        "turn_based": env.is_turn_based,
                        "penalty": FLAGS.penalty,
                        "data": data, 
                        "session": sess,
                        "indicator_rounding":FLAGS.indicator_rounding,
                        "reward_structure": reward_structure}

            model = WorldModelDeterministic(state_size, action_size, model_args)
            losses = model.train(num_gradient_steps=10000)
            print("Model Training finished.")

            plt.plot(range(len(losses)), losses)
            plt.savefig('model_loss_before_training.jpg')

            response_parameters = {
                "state_representation_size": env._game.information_state_tensor_shape()[0], 
                "num_actions": env.action_spec()["num_actions"],
                "double": FLAGS.double, 
                "hidden_layers_sizes": [FLAGS.dqn_model_width] * FLAGS.dqn_model_depth, 
                "replay_buffer_capacity": FLAGS.replay_buffer_size, 
                "batch_size": FLAGS.dqn_batch_size, 
                "learning_rate": FLAGS.dqn_learning_rate,
                "update_target_network_every": FLAGS.update_target_every, 
                "learn_every": FLAGS.learn_every, 
                "discount_factor": FLAGS.discount, 
                "min_buffer_size_to_learn": FLAGS.min_buffer_size_learn, 
                "epsilon_start": FLAGS.epsilon_start, 
                "epsilon_end": FLAGS.epsilon_end, 
                "epsilon_decay_duration": FLAGS.epsilon_decay_duration 
            }

            # Instantiate DDQN 
            training_policy = DQN(sess, graph=rl_training_graph, **response_parameters)
            response_target = UniformRandomPolicy(env.action_spec()["num_actions"], env._game.information_state_tensor_shape()[0])
            policies = [training_policy, response_target]
            
            num_training_steps = 0
            episodes = 0
            model_returns_aggregate = []
            returns_aggregate, num_training_steps, episodes = [], 0, 0
            print("Beginning DDQN Training")
            while(num_training_steps < FLAGS.num_training_steps):
                # Sample from the trajectories to find a starting point
                sampled_start = np.random.choice(trajectories)[0]
                
                global_state = sampled_start.global_state
                relevant_players = sampled_start.relevant_players
                curr_model_returns = 0
                if env.is_turn_based:
                    all_observations = [[sampled_start.info_states[relevant_players[0]]] for p in range(env.num_players)]  # relevant_players[0] is just the first player to go
                    info_state = true_state_extractor.observations_to_info_state(all_observations[relevant_players[0]])
                    curr_step = Step(info_state=info_state, reward=None, is_terminal=False, legal_actions_mask=sampled_start.legal_actions_masks[relevant_players[0]],
                                    acting_players=relevant_players, global_state=global_state)
                    
                else:
                    # all_observations = [[curr_step.info_state[p]] for p in env.num_players]
                    raise NotImplementedError

                done = False 
                steps_so_far = 0
                while not done:
                    if env.is_turn_based:
                        curr_player = curr_step.acting_players[0]
                        action = [policies[curr_player].step(curr_step, curr_player, session=sess, is_evaluation=False)[0]]

                        next_state, reward, player_next_observations, next_legal_action_masks, done, halt = model.get_next_step(curr_step.global_state, action, FLAGS.halt_threshold)
                        steps_so_far += 1
                        done = done or steps_so_far >= FLAGS.max_episode_length
                        curr_model_returns += reward[0][0]

                        next_player = (relevant_players[0] + 1) % env.num_players # NOTE: this is an assumption we are making! 

                        for p in range(env.num_players):
                            all_observations[p].append(player_next_observations[next_player])

                        # Construct the next step for the upcoming player
                        info_state = true_state_extractor.observations_to_info_state(all_observations[next_player])
                        
                        # Reassign curr_step. If we are "done," then only reward is used.
                        curr_step = Step(info_state=info_state, reward=reward, is_terminal=done, legal_actions_mask=next_legal_action_masks, acting_players=[next_player], global_state=next_state)
                    else:
                        raise NotImplementedError

                    num_training_steps += 1
                for p in range(env.num_players):
                    policies[p].step(curr_step, p, session=sess, is_evaluation=False)  # last time step account for terminal
                episodes += 1
                model_returns_aggregate.append(curr_model_returns)

                if episodes % 100 == 0:
                    
                    evaluation_returns = []
                    num_episodes_evaluate = 200
                    for _ in range(num_episodes_evaluate):
                        rollout = generate_single_rollout(env.num_players, env, policies, sessions=[sess, None], is_turn_based=env.is_turn_based, true_state_extractor=true_state_extractor)
                        ret = sum([t.rewards[0] for t in rollout])
                        evaluation_returns.append(ret)
                    evaluation = sum(evaluation_returns) / num_episodes_evaluate
                    returns_aggregate.append(evaluation)
                    print("Evaluation after {} episodes or {} training steps: {}".format(episodes, num_training_steps, evaluation))

            print("\n\nEntering final evaluation for 2000 episodes: ")
            evaluation_returns = []
            num_episodes_evaluate = 2000
            for _ in range(num_episodes_evaluate):
                rollout = generate_single_rollout(env.num_players, env, policies, sessions=[sess, None], is_turn_based=env.is_turn_based, true_state_extractor=true_state_extractor)
                ret = sum([t.rewards[0] for t in rollout])
                evaluation_returns.append(ret)
            evaluation = sum(evaluation_returns) / num_episodes_evaluate
            print("Final evaluation estimated to be: {}".format(evaluation))
            returns_aggregate.append(evaluation)
                    
            plt.clf()
            window = 50
            averaged_returns = [sum(returns_aggregate[i: i + window]) / window for i in range(len(returns_aggregate) - window + 1)]
            plt.plot(range(len(averaged_returns)), averaged_returns)
            # plt.title("Round 1 Tuning: Target {} Learn Every {} Batch {}".format(FLAGS.update_target_every, FLAGS.learn_every, FLAGS.dqn_batch_size))

            save_path = FLAGS.save_path if FLAGS.save_path[-1] == "/" else FLAGS.save_path + "/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(save_path+"evaluation_returns.jpg")

            print("\n\nModel returns aggregate: ", model_returns_aggregate)

    if FLAGS.experiment_name == "respond_to_uniform_mixture":
        directories = ["player_0/"] if FLAGS.symmetric else ["player_{}/".format(p) for p in range(env._num_players)]
        all_frozen_graphs, context_managers, variable_names = get_graphs_and_context_managers(FLAGS.load_path, directories, TFModelManagement())

        training_player = 0
        profile_respond_to = [[1.0 / len(all_frozen_graphs[0]) for _ in all_frozen_graphs[0]] for _ in range(env._num_players)]
        
        response_parameters = {
            "state_representation_size": env._game.information_state_tensor_shape()[0], 
            "num_actions": env.action_spec()["num_actions"],
            "double": FLAGS.double, 
            "hidden_layers_sizes": [FLAGS.dqn_model_width] * FLAGS.dqn_model_depth, 
            "replay_buffer_capacity": FLAGS.replay_buffer_size, 
            "batch_size": FLAGS.dqn_batch_size, 
            "learning_rate": FLAGS.dqn_learning_rate,
            "update_target_network_every": FLAGS.update_target_every, 
            "learn_every": FLAGS.learn_every, 
            "discount_factor": FLAGS.discount, 
            "min_buffer_size_to_learn": FLAGS.min_buffer_size_learn, 
            "epsilon_start": FLAGS.epsilon_start, 
            "epsilon_end": FLAGS.epsilon_end, 
            "epsilon_decay_duration": FLAGS.epsilon_decay_duration 
        }
        
        with ExitStack() as stack:
            # Load policy graphs 
            sessions = [[stack.enter_context(cm) if cm != None else None for cm in cm_list] for cm_list in context_managers]

            if FLAGS.symmetric:
                all_frozen_graphs = [all_frozen_graphs[0] for _ in range(env._num_players)]
                context_managers = [context_managers[0] for _ in range(env._num_players)]
                variable_names = [variable_names[0] for _ in range(env._num_players)]
                sessions = [sessions[0] for _ in range(env._num_players)]

            # Initialize all of the profiles using DQN and pass in respective sessions/graphs
            all_policies = []
            for p in range(len(sessions)):
                curr_policy_list = []
                for s in range(len(sessions[p])):
                    curr_policy = DQN(sessions[p][s], graph=all_frozen_graphs[p][s], start_frozen=True, **response_parameters)
                    curr_policy.load_variable_names(variable_names[p][s])
                    curr_policy_list.append(curr_policy)
                all_policies.append(curr_policy_list)

            # Create a new computational graph for our new policy
            new_rl_training_session = stack.enter_context(tf.Session(graph=tf.Graph()))
            sessions[training_player].append(new_rl_training_session)

            # Initialize new policy and append
            new_policy = DQN(new_rl_training_session, graph=new_rl_training_session.graph, **response_parameters)

            # Train model
            if FLAGS.game_name == "bargaining":
                indicator_rounding, reward_structure = True, "terminal"
            else:
                raise NotImplementedError

            model_args = {"hidden_sizes": [FLAGS.model_width] * FLAGS.model_depth,
                        "batch_size": FLAGS.batch_size,
                        "learning_rate": FLAGS.learning_rate,
                        "ensemble_size": FLAGS.ensemble_size,
                        "observation_sizes":[env._game.observation_tensor_shape()[0]] * env.num_players,
                        "num_players":env.num_players,
                        "num_actions":env.action_spec()["num_actions"],
                        "turn_based": env.is_turn_based,
                        "penalty": FLAGS.penalty,
                        "data": data, 
                        "session": new_rl_training_session,
                        "indicator_rounding": indicator_rounding,
                        "reward_structure": reward_structure}

            model = WorldModelDeterministic(state_size, action_size, model_args)
            losses = model.train(num_gradient_steps=10000)
            print("Model Training finished.")

            # Insert training loop here. Use an env and manual step creations to provide learning to the currently training response
            num_training_steps = 0
            episodes = 0
            model_returns_aggregate = []
            returns_aggregate, num_training_steps, episodes = [], 0, 0
            print("Beginning DDQN Training")
            while num_training_steps < FLAGS.num_training_steps:
                # Sample from profile and then create the current sessions being used by the respective players
                curr_sessions = []
                policies = []
                for player in range(env._num_players):
                    if player != training_player:
                        choice = np.random.choice(range(len(profile_respond_to[player])), p=profile_respond_to[player])
                        curr_sessions.append(sessions[player][choice])
                        policies.append(all_policies[player][choice])
                    else:
                        curr_sessions.append(new_rl_training_session)
                        policies.append(new_policy)

                # Sample from the trajectories to find a starting point
                sampled_start = np.random.choice(trajectories)[0]
                
                global_state = sampled_start.global_state
                relevant_players = sampled_start.relevant_players
                curr_model_returns = 0
                if env.is_turn_based:
                    all_observations = [[sampled_start.info_states[relevant_players[0]]] for p in range(env.num_players)]  # relevant_players[0] is just the first player to go
                    info_state = true_state_extractor.observations_to_info_state(all_observations[relevant_players[0]])
                    curr_step = Step(info_state=info_state, reward=None, is_terminal=False, legal_actions_mask=sampled_start.legal_actions_masks[relevant_players[0]],
                                    acting_players=relevant_players, global_state=global_state)
                    
                else:
                    # all_observations = [[curr_step.info_state[p]] for p in env.num_players]
                    raise NotImplementedError

                done = False 
                steps_so_far = 0
                while not done:
                    if env.is_turn_based:
                        curr_player = curr_step.acting_players[0]
                        action = [policies[curr_player].step(curr_step, curr_player, session=curr_sessions[curr_player], is_evaluation=False)[0]]

                        next_state, reward, player_next_observations, next_legal_action_masks, done, halt = model.get_next_step(curr_step.global_state, action, FLAGS.halt_threshold)
                        steps_so_far += 1
                        done = done or steps_so_far >= FLAGS.max_episode_length

                        if halt:
                            reward = [[r[0] - FLAGS.penalty] for r in reward]

                        curr_model_returns += reward[0][0]

                        next_player = (relevant_players[0] + 1) % env.num_players # NOTE: this is an assumption we are making! 

                        # Only if the episode is not done do we append new observations (otherwise it's None) and generate a new info_state
                        # if not done:
                        # Append new observations 
                        for p in range(env.num_players):
                            all_observations[p].append(player_next_observations[next_player])

                        # Construct the next step for the upcoming player
                        info_state = true_state_extractor.observations_to_info_state(all_observations[next_player])
                        
                        # Reassign curr_step. If we are "done," then only reward is used.
                        curr_step = Step(info_state=info_state, reward=reward, is_terminal=done, legal_actions_mask=next_legal_action_masks, acting_players=[next_player], global_state=next_state)
                    else:
                        raise NotImplementedError

                    num_training_steps += 1
                for p in range(env.num_players):
                    policies[p].step(curr_step, p, session=curr_sessions[p], is_evaluation=False)  # last time step account for terminal

                episodes += 1
                model_returns_aggregate.append(curr_model_returns)
            print("Total episodes trained on: ", len(model_returns_aggregate))
            print("Model training final returns: ", np.mean(model_returns_aggregate[-2000: ]))

            # print("\n\nEntering final evaluation for 2000 episodes: ")
            # evaluation_returns = []
            # num_episodes_evaluate = 2000
            # for _ in range(num_episodes_evaluate):
            #     curr_sessions = []
            #     policies = []
            #     for player in range(env._num_players):
            #         if player != training_player:
            #             choice = np.random.choice(range(len(profile_respond_to[player])), p=profile_respond_to[player])
            #             curr_sessions.append(sessions[player][choice])
            #             policies.append(all_policies[player][choice])
            #         else:
            #             curr_sessions.append(new_rl_training_session)
            #             policies.append(new_policy)
            #     rollout = generate_single_rollout(env.num_players, env, policies, sessions=curr_sessions, is_turn_based=env.is_turn_based, true_state_extractor=true_state_extractor)
            #     ret = sum([t.rewards[training_player] for t in rollout])
            #     evaluation_returns.append(ret)
            # evaluation = sum(evaluation_returns) / num_episodes_evaluate
            # print("Final evaluation estimated to be: {}".format(evaluation))
            # returns_aggregate.append(evaluation)
                    
    print("Finished experiment.")
    


if __name__ == "__main__":
  app.run(main)
