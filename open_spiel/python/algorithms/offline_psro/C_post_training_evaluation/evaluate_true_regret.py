"""
This file is dedicated to evaluating the TRUE game regret of intermediate profiles for a given offline PSRO run.

Instead of computing a true game best response for each iteration profile for all offline PSRO runs, we are going to construct a fixed
set of strategies to evaluate true game. This set of strategies is going to consist of 1) strategies uncovered in the current offline PSRO trial and
2) TRUE game best responses to chosen profiles across all offline PSRO runs (these will typically be equilibria found by offline PSRO)

We do NOT reconstruct empirical games for each of the iterations. Instead, we assimilate all strategies to create our fixed evaluation set.
Then, for each profile we are attempting to evaluate regret, we cycle through each of the strategies in the strategy set for a best response 
by using the simulator. We calculate regret using sampling.
"""


from absl import app
from absl import flags
import numpy as np
import os
import matplotlib.pyplot as plt
from absl import logging
import pickle 
import tensorflow.compat.v1 as tf
import pyspiel
import itertools 
import time

from contextlib import ExitStack 
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.world_model_deterministic import WorldModelDeterministic
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.ddqn_simple import DQN
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.uniform_random_policy import UniformRandomPolicy
from open_spiel.python.algorithms.offline_psro.B_training.tf_model_management.tf_model_management import TFModelManagement
from open_spiel.python.rl_environment import StepType
from open_spiel.python.algorithms.offline_psro.utils.utils import generate_single_rollout, Step, get_graphs_and_context_managers, Normalizer
from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.bargaining_true_state_generator import BargainingTrueStateGenerator
from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.bargaining_generalized_true_state_generator import BargainingGeneralizedTrueStateGenerator
from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.leduc_poker_true_state_generator import LeducPokerTrueStateGenerator
from open_spiel.python import rl_environment
from scipy import stats 

from open_spiel.python.algorithms import projected_replicator_dynamics

FLAGS = flags.FLAGS 

# Simulation related
flags.DEFINE_bool("symmetric", False, "Is the game symmetric?")
flags.DEFINE_integer("num_players", 2, "Number of players in the game")
flags.DEFINE_string("save_path", "random_runs/", "Folder to where all saved models and data live")
flags.DEFINE_string("save_graph_path", "random_runs/", "Folder to save all NEW models and data")
flags.DEFINE_integer('num_simulations', 1000, "Number of simulations to estimate normal-form game entries")
flags.DEFINE_integer("num_bootstrapped_samples", 1000, "Number of bootstrapped samples when estimating expected regret")
flags.DEFINE_string("evaluation_strategy_path", "", "Folder to save or load any evaluation set strategies")
flags.DEFINE_string("game_name", "", "Name of game")
flags.DEFINE_bool("eval_regret_with_final_profiles", False, "Whether to use the extracted final profiles at each iteration to calculate regret, not the MSS used during the trial")
flags.DEFINE_string("dataset_path", "", "Path to file to load the offline dataset")

flags.DEFINE_string("evaluation_name", "evaluate_regret", "Whether to train a new evaluation strategy or evaluate reget of a trial")

# Misc
flags.DEFINE_integer("seed", 1, "Seed for random")

def get_manual_chance_actions(game, count):
    if game == "bargaining":
        return list(np.random.choice(31250, count))
    else:
        return []


class TrueGameRegretEvaluationModule:
    def __init__(self, env, num_players, evaluation_strategy_path, save_data_path, symmetric, num_simulations, true_state_extractor):
        """
        env: rl_environment object to interact with the true game
        init_policies: list of policies (one per player) that describes the initial policies used in the trial
        policy_type: string describing what kind of policy was trained for each of the added strategies for the trial
        save_data_path: relative path to where we will save any evaluation data
        """
        self._env = env
        self._evaluation_strategy_path = evaluation_strategy_path # This should be player specific.
        self._save_data_path = save_data_path if save_data_path[-1] == "/" else save_data_path + "/"
        self._tf_model_manager = TFModelManagement()
        self._true_state_extractor = true_state_extractor
        self._num_players = num_players 
        self._symmetric = symmetric
        self._num_simulations = num_simulations

    def load_profile(self, trial_data_path, profile_file_name):
        with open(trial_data_path + profile_file_name, 'rb') as f:
            profile_respond_to = pickle.load(f)
        return profile_respond_to

    def train_true_game_best_response(self, trial_data_path, player_trial_strategy_path, profile_file_name, training_player, training_steps_total, response_parameters):
        """
        trial_data_path: path to folder containing empirical game and profile information at each iteration of a trial
        player_trial_strategy_path: a list containing paths to folders of each player's strategies created during that trial 
        profile_file_name: the name of the particular file in trial_data_path containing the profile we want to respond to
        """
        # NOTE: trial_data_path contains information on profiles. player_trial_strategy_path describes the folder containing player-specific policies.
        if self._symmetric:
            assert len(player_trial_strategy_path) == 1
        else:
            assert len(player_trial_strategy_path) == self._env._num_players

        # Load the profile we are responding to 
        profile_respond_to = self.load_profile(trial_data_path, profile_file_name)
        print("\nResponding to profile: ", profile_respond_to, "  \n")

        all_frozen_graphs, context_managers, variable_names = get_graphs_and_context_managers(trial_data_path, player_trial_strategy_path, self._tf_model_manager)

        print("Training new strategy. ")
        # Load all the sessions and graphs from the previously frozen policies
        with ExitStack() as stack:
            # Load policy graphs 
            sessions = [[stack.enter_context(cm) if cm != None else None for cm in cm_list] for cm_list in context_managers]

            # Initialize all of the profiles using DQN and pass in respective sessions/graphs and variable names
            all_policies = []
            for p in range(len(sessions)):
                curr_policy_list = []
                for s in range(len(sessions[p])):
                    curr_policy = DQN(sessions[p][s], graph=all_frozen_graphs[p][s], start_frozen=True, **response_parameters)
                    curr_policy.load_variable_names(variable_names[p][s])
                    curr_policy_list.append(curr_policy)
                all_policies.append(curr_policy_list)

                
            # TODO: Add initial strategies here
            for p in range(len(context_managers)):
                all_policies[p].insert(0, UniformRandomPolicy(state_size=self._env._game.information_state_tensor_shape()[0], 
                                                            num_actions=self._env.action_spec()["num_actions"]))
                sessions[p].insert(0, None)

                assert len(all_policies[p]) >= len(profile_respond_to[p])

            # Create a new computational graph for our new policy
            new_rl_training_session = stack.enter_context(tf.Session(graph=tf.Graph()))
            sessions[training_player].append(new_rl_training_session)

            # Initialize new policy and append
            new_policy = DQN(new_rl_training_session, graph=new_rl_training_session.graph, reward_normalizer=all_policies[0][-1]._reward_normalizer, **response_parameters)

            # If symmetric, then all players share policies
            if FLAGS.symmetric and len(all_policies) == 1:
                all_policies = [all_policies[0] for _ in range(FLAGS.num_players)]
                sessions = [sessions[0] for _ in range(FLAGS.num_players)]

            # Training loop. Use an rl_environment and manual step creations to provide learning to the currently training response
            aggregate_rewards = []
            num_training_steps = 0

            while num_training_steps < training_steps_total:
                # Sample from profile and then create the current sessions being used by the respective players
                curr_sessions = []
                policies = []
                for player in range(self._env._num_players):
                    if player != training_player:
                        choice = np.random.choice(range(len(profile_respond_to[player])), p=profile_respond_to[player])
                        curr_sessions.append(sessions[player][choice])
                        policies.append(all_policies[player][choice])
                    else:
                        curr_sessions.append(new_rl_training_session)
                        policies.append(new_policy)
                
                timestep = self._env.reset()
                if self._env.is_turn_based:
                    curr_player = timestep.observations["current_player"]
                    curr_step = Step(info_state=timestep.observations["info_state"][curr_player], reward=None, is_terminal=False, 
                                    legal_actions_mask=[1 if a in timestep.observations["legal_actions"][curr_player] else 0 for a in range(self._env.action_spec()["num_actions"])],
                                    acting_players=[curr_player], global_state=None)
                else:
                    raise NotImplemented
                
                done, returns = False, 0
                
                while not done:
                    if self._env.is_turn_based:
                        # Get agent actions
                        curr_player = curr_step.acting_players[0]

                        # Track how many steps training_player has taken
                        if curr_player == training_player:
                            num_training_steps += 1

                        action = [policies[curr_player].step(curr_step, curr_player, session=curr_sessions[curr_player], is_evaluation=False)[0]]

                        # Apply action to env
                        timestep = self._env.step(action)
                        returns += timestep.rewards[training_player]

                        # Check if done
                        done = timestep.step_type == StepType.LAST

                        # Convert timestep to Step
                        if not done:
                            curr_player = timestep.observations["current_player"]
                            curr_step = Step(info_state=timestep.observations["info_state"][curr_player], reward=[timestep.rewards], is_terminal=done,
                                            legal_actions_mask=[1 if a in timestep.observations["legal_actions"][curr_player] else 0 for a in range(self._env.action_spec()["num_actions"]) ],
                                            acting_players=[curr_player], global_state=None)
                    else:
                        raise NotImplementedError
                
                # Last time step to account for terminal state
                for p in range(self._num_players):
                    curr_step = Step(info_state=timestep.observations["info_state"][p], reward=[timestep.rewards], is_terminal=done,
                                            legal_actions_mask=[1 if a in timestep.observations["legal_actions"][p] else 0 for a in range(self._env.action_spec()["num_actions"]) ],
                                            acting_players=[p], global_state=None)
                    policies[p].step(curr_step, p, curr_sessions[p], is_evaluation=False)  
                aggregate_rewards.append(returns)

            window = 200
            averaged_over = [sum(aggregate_rewards[i:i+window]) / window for i in range(len(aggregate_rewards) - window + 1)]
            plt.plot(list(range(len(averaged_over))), averaged_over)

            # To distinguish between different players, we create player-specific directories for their policies.
            # In symmetric games, there will only be a folder for player 0 (because strategies are shared)
            policy_save_path = self._evaluation_strategy_path + "player_{}/".format(training_player)
            if not os.path.exists(policy_save_path):
                os.makedirs(policy_save_path)
            
            num_policies_so_far = len([f for f in os.listdir(policy_save_path) if ("evaluation_policy" in f) and (".ckpt.data" in f)])

            plt.savefig(policy_save_path+'policy_model_rewards_policy_{}_player_{}.jpg'.format(num_policies_so_far, training_player))
            plt.clf()
            
            # After training, save the newly trained response to self._evaluation_strategy_path
            self._tf_model_manager.save(new_rl_training_session, policy_save_path + 'evaluation_policy_{}.ckpt'.format(num_policies_so_far))
    
        # Outside of the session, freeze and load the newly trained policy. 
        new_policy._id = num_policies_so_far
        new_policy.freeze(self._tf_model_manager, policy_save_path)
        
        return new_policy

    def evaluate_regret(self, trial_data_path, player_trial_strategy_path, final=False):
        # TODO: Use the aggregate trial_data_path along with our evaluation_path to our evaluation_strategy_set to simulate multiple times what the best response might actually be

        num_strategies_each_player = []
        for player_path in player_trial_strategy_path:
            num_strategies = len([f for f in os.listdir(trial_data_path + player_path) if ("policy" in f) and (".pb" in f)]) + 1
            num_eval_strategies = len([f for f in os.listdir(self._evaluation_strategy_path + player_path) if "policy" in f and ".pb" in f])
            num_strategies_each_player.append(num_strategies + num_eval_strategies)
        
        if self._symmetric:
            num_strategies_each_player = num_strategies_each_player * self._num_players

        # Reconstruct the entire empirical game (we are bound to use each of the entries at least once so it makes sense!) but with true game utilities
        empirical_game = {}

        # Combine the policies in trial_data_path/player_trial_strategy_path and self._evaluation_strategy_set. This will be our aggregate evaluation set
        evaluation_graphs, evaluation_context_managers, evaluation_variable_names = get_graphs_and_context_managers(trial_data_path, player_trial_strategy_path, self._tf_model_manager)
        additional_graphs, additional_context_managers, additional_variable_names = get_graphs_and_context_managers(self._evaluation_strategy_path, player_trial_strategy_path, self._tf_model_manager)
        
        for i in range(len(additional_graphs)):
            evaluation_graphs[i].extend(additional_graphs[i])
            evaluation_context_managers[i].extend(additional_context_managers[i])
            evaluation_variable_names[i].extend(additional_variable_names[i])

        with ExitStack() as stack:
            # Load policy graphs 
            sessions = [[stack.enter_context(cm) if cm != None else None for cm in cm_list] for cm_list in evaluation_context_managers]

            # Initialize all of the profiles using DQN and pass in respective sessions/graphs and variable names
            all_policies = []
            for p in range(len(sessions)):
                curr_policy_list = []
                for s in range(len(sessions[p])):
                    curr_policy = DQN(sessions[p][s], graph=evaluation_graphs[p][s], start_frozen=True, 
                                        state_representation_size=self._env._game.information_state_tensor_shape()[0], 
                                        num_actions=self._env.action_spec()["num_actions"],)
                    curr_policy.load_variable_names(evaluation_variable_names[p][s])
                    curr_policy_list.append(curr_policy)
                # Insert uniformActionPolicy here
                curr_policy_list.insert(0, UniformRandomPolicy(state_size=self._env._game.information_state_tensor_shape()[0], 
                                                            num_actions=self._env.action_spec()["num_actions"]))
                sessions[p].insert(0, None)

                all_policies.append(curr_policy_list)
            
            if self._symmetric:
                all_policies = [all_policies[0] for _ in range(self._num_players)]
                sessions = [sessions[0] for _ in range(self._num_players)]    

            strategy_choices = [list(range(num_strategies)) for num_strategies in num_strategies_each_player]
            print("Reconstructing empirical game. ")
            for chosen_strategies in itertools.product(*strategy_choices):
                all_returns = []

                # If there is no entry 
                if tuple(chosen_strategies) not in empirical_game:

                    # i should be the player. j should be the strategy player i has chosen
                    policies = [all_policies[i][j] for i, j in enumerate(chosen_strategies)]
                    policy_sessions = [sessions[i][j] for i, j in enumerate(chosen_strategies)]

                    # Simulation loop
                    start = time.time() 

                    for _ in range(self._num_simulations):
                        rollout = generate_single_rollout(self._num_players, self._env, policies, policy_sessions, self._env.is_turn_based, true_state_extractor=self._true_state_extractor)
                        returns = 0
                        for t in rollout:
                            returns += np.array(t.rewards)
                        all_returns.append(returns)
                    print("Entry {} took {} seconds to fill with {} simulations. ".format(chosen_strategies, time.time() - start, self._num_simulations))

                    all_returns = np.array(all_returns)
                    empirical_game[tuple(chosen_strategies)] = np.mean(all_returns, axis=0)

                    # If symmetric, then these utilities can be used for all permutations of current player policies. Utilities need to be permuted accordingly
                    if self._symmetric:
                        all_symmetric_entries = itertools.permutations(chosen_strategies)

                        for entry in all_symmetric_entries:
                            swapped_returns = []
                            for s in entry:
                                swapped_returns.append(all_returns[:, chosen_strategies.index(s)])
                            empirical_game[tuple(entry)] = np.mean(np.array(swapped_returns).T, axis=0)

        print("Empirical game constructed. Now calculating regret. ")
        # Using the empirical game, extract the regret of each profile in trial_data_path
        # Get all profiles from trial_data_path
        all_profile_files = [f for f in os.listdir(trial_data_path) if (("profile" in f) if not final else "solution" in f) and (".pkl" in f) and ("final" in f if final else "final" not in f)]
        all_profile_files = sorted(all_profile_files, key=lambda s: int(s.split('_')[-1].split('.')[0]))
        print("All profile files: ", all_profile_files)
        all_profiles = []
        for f in all_profile_files:
            with open(trial_data_path+f, 'rb') as f:
                all_profiles.append(pickle.load(f))
        
        print("Profile 1: ", all_profiles)

        all_profiles.insert(0, [np.array([1]) for p in range(self._num_players)])

        regret_values = [[] for _ in range(self._num_players)]

        for profile in all_profiles:
            # This is a minute but important point: we take [1:] of the player profile because we don't save the initial, singleton policy in PSRO
            profile = [np.array(player_profile) for player_profile in profile]
            profile = [np.pad(player_profile, (0, num_strategies_each_player[player] - player_profile.shape[0])) for player, player_profile in enumerate(profile)]

            for player, player_meta_strategy in enumerate(profile):
                deviating_payoffs = np.zeros(num_strategies_each_player[player])
                curr_payoff = 0
                for chosen_strategies in itertools.product(*strategy_choices):
                    player_strategy = chosen_strategies[player]
                    prob = np.prod([profile[index][chosen_strategies[index]] for index in range(self._num_players)]) 
                    prob_others = np.prod([profile[index][chosen_strategies[index]] for index in range(self._num_players) if index != player])
                    payoff = empirical_game[chosen_strategies][player]

                    curr_payoff += prob * payoff
                    deviating_payoffs[player_strategy] += prob_others * payoff
                
                print("Best deviating strategy: ", np.argmax(deviating_payoffs))
                player_regret = np.max(deviating_payoffs) - curr_payoff
                regret_values[player].append(player_regret)
        
        print("Regret values: ", regret_values)
        print("Empirical_game: ", empirical_game)
    
        with open(trial_data_path+"true_game_regret_calculations.pkl", 'wb') as f:
            pickle.dump(regret_values, f)
        
        with open(trial_data_path+"true_game_empirical_game.pkl", 'wb') as f:
            pickle.dump(empirical_game, f)

        for p, regret_trend in enumerate(regret_values):
            plt.plot(range(len(regret_trend)), regret_trend)
            plt.savefig(trial_data_path+"regret_trends_player_{}.jpg".format(p))
            plt.clf()        

        return empirical_game, regret_values

    def evaluate_trajectory_level_reward_uncertainty_correlations(self, pyspiel_game, trial_data_path, player_trial_strategy_path, true_state_extractor, save_graph_path, response_parameters):
        
        with open(FLAGS.dataset_path, "rb") as npy_file:
            trajectories = np.load(npy_file, allow_pickle=True)
        data = []

        for rollout in trajectories:
            data.extend(rollout)
        
        env = rl_environment.Environment(pyspiel_game, observation_type=rl_environment.ObservationType.OBSERVATION)

        action_size = 1 if env.is_turn_based else FLAGS.num_players

        # Get the model graph
        tf_model_management_module = TFModelManagement()
        model_graph = tf_model_management_module.load_frozen_graph(trial_data_path, WorldModelDeterministic.frozen_graph_model_name)
        
        # Load the dynamics model along with the graphs 
        # Dynamics model parameters
        with open(trial_data_path + WorldModelDeterministic.model_arg_name, 'rb') as f:
            model_args = pickle.load(f)
            model_args["data"] = None 
            model_args["session"] = None

        model = WorldModelDeterministic(state_size=len(data[0].global_state), action_size=action_size, model_args=model_args,graph=model_graph, start_frozen=True)
        with open(trial_data_path + model.get_variable_name_file_name(), 'rb') as f:
            model_variables = pickle.load(f)
        
        model.load_variable_names(model_variables)

        # Initializations 
        model_context = tf.Session(graph=model.get_frozen_graph())

        # For each of entries in our empirical game, run the joint pure strategy in the real game 
        # Get the strategies from our directory 
        evaluation_graphs, evaluation_context_managers, evaluation_variable_names = get_graphs_and_context_managers(trial_data_path, player_trial_strategy_path, self._tf_model_manager)
        with ExitStack() as stack: 
            all_policy_sessions = [[stack.enter_context(cm) if cm != None else None for cm in cm_list] for cm_list in evaluation_context_managers]
            model_session = stack.enter_context(model_context)

            all_policies = []
            for p in range(len(all_policy_sessions)):
                curr_policy_list = []
                for s in range(len(all_policy_sessions[p])):
                    curr_policy = DQN(all_policy_sessions[p][s], graph=evaluation_graphs[p][s], start_frozen=True, **response_parameters)
                    curr_policy.load_variable_names(evaluation_variable_names[p][s])
                    curr_policy_list.append(curr_policy)
                all_policies.append(curr_policy_list)

            for p in range(len(all_policy_sessions)):
                all_policy_sessions[p].insert(0, None)
                all_policies[p].insert(0, UniformRandomPolicy(state_size=env._game.information_state_tensor_shape()[0],num_actions=env.action_spec()["num_actions"]))
            
            
             # If symmetric, then all players share policies
            if FLAGS.symmetric and len(all_policies) == 1:
                all_policies = [all_policies[0] for _ in range(FLAGS.num_players)]
                all_policy_sessions = [all_policy_sessions[0] for _ in range(FLAGS.num_players)]

            all_discrepancies, all_reward_errors, all_utility_errors = {}, [], {}

            all_pure_strategies = itertools.product(*[list(range(len(policy_list))) for policy_list in all_policies])

            for pure_strategy in all_pure_strategies:
                if not all([len(all_policies[p]) > pure_strategy[p] for p in range(self._num_players)]):
                    continue 
                num_trajectories = int(1e4)
                terminal_reward_errors = []
                discrepancy_reward_predictions = []
                all_true_terminal_rewards, all_model_terminal_rewards = [], []
                policies = [all_policies[p][strategy] for p, strategy in enumerate(pure_strategy)]
                policy_sessions = [all_policy_sessions[p][strategy] for p, strategy in enumerate(pure_strategy)]

                start = time.time()
                # manual_samples = get_manual_chance_actions(FLAGS.game_name, num_trajectories)

                for traj_index in range(num_trajectories):
                    sample = None # manual_samples[traj_index]
                    
                    traj = generate_single_rollout(FLAGS.num_players, env, policies, policy_sessions, env.is_turn_based, true_state_extractor, manually_create_info_states_from_observations=True, start_action=sample)

                    for transition in traj[-1:]:
                        # Only look at the terminal reward for now
                        if transition.done:
                            _, reward, _, _, _, _, max_prediction_discrepancy = model.get_next_step(transition.global_state, [a for a in transition.actions if a != None], halt_threshold=10, frozen_session=model_session)

                            curr_reward_error = np.sum(np.abs(np.array(transition.rewards) - np.array(reward[0])))
                            
                            if transition.done:
                                terminal_reward_errors.append(curr_reward_error)
                                discrepancy_reward_predictions.append(max_prediction_discrepancy)

                                all_true_terminal_rewards.append(transition.rewards)
                                all_model_terminal_rewards.append(reward[0])
                simulation_time = time.time() - start 

                start = time.time() 
                # Calculate correlation coefficient 
                r = stats.pearsonr(terminal_reward_errors, discrepancy_reward_predictions)
                analysis_time = time.time() - start
                all_discrepancies[pure_strategy] = sum(discrepancy_reward_predictions) / len(discrepancy_reward_predictions)
                all_reward_errors.append(sum(terminal_reward_errors) / len(terminal_reward_errors))
                utility_error = np.mean(np.abs(np.mean(np.array(all_true_terminal_rewards), axis=0) - np.mean(np.array(all_model_terminal_rewards), axis=0)))
                all_utility_errors[pure_strategy] = utility_error
                
                print("Pure strategy: {}    R: {}   Simulation time: {}     Analysis time: {}   Error: {}   Discrepancy: {}     Utility Error: {}   True Utility: {}".format(pure_strategy, r, simulation_time, analysis_time, sum(terminal_reward_errors) / len(terminal_reward_errors), sum(discrepancy_reward_predictions) / len(discrepancy_reward_predictions), utility_error, np.mean(np.array(all_true_terminal_rewards), axis=0)))

                if not os.path.exists(save_graph_path):
                    os.makedirs(save_graph_path)
                plt.scatter(terminal_reward_errors, discrepancy_reward_predictions, s=1)
                plt.title("reward errors and discrepancies trajectory level")
                plt.savefig(save_graph_path+"reward_errors_discrepancy_strategy_{}.jpg".format(pure_strategy))
                plt.clf()

                plt.hist(terminal_reward_errors, bins=np.arange(160)/4.0)
                plt.title("reward error frequencies")
                plt.savefig(save_graph_path+"reward_errors_histogram_strategy_{}.jpg".format(pure_strategy))
                plt.clf()

            print("Average discrepancy across all pure strategies: {}".format(sum(all_discrepancies.values()) / len(all_discrepancies)))
            print("Average reward errors across all pure strategies: {}".format(sum(all_reward_errors) / len(all_reward_errors)))
            print("Average utility error across all pure strategies: {}".format(sum(all_utility_errors.values()) / len(all_utility_errors)))

        reward_error_information = {"all_reward_errors": all_reward_errors, "all_discrepancies": all_discrepancies, "all_utility_errors": all_utility_errors}
        with open(trial_data_path+"reward_error_information.pkl", 'wb') as f:
            pickle.dump(reward_error_information, f)

        return 
    
    def analyze_deviation_coverage_by_iteration(self, num_players, trial_data_path, iteration_profiles):

        with open(trial_data_path+"reward_error_information.pkl", 'rb') as f:
            reward_error_information = pickle.load(f)
        
        all_discrepancies = reward_error_information["all_discrepancies"]
        all_utility_errors = reward_error_information["all_utility_errors"]

        all_profile_utility_errors = []
        all_deviation_coverage_metrics = []
        
        num_strategies_per_player = []
        for p in range(num_players):
            num_strategies_per_player.append(max([s[p] + 1 for s in all_discrepancies.keys()]))
        
        for profile_name in iteration_profiles:
            with open(trial_data_path + profile_name, 'rb') as f:
                current_profile = pickle.load(f)

            print("Analyzing profile: ", current_profile)

            # current_profile is a length num_player vector, where each element is a list describing player's meta-strategy

            # Get the cartesian product of each of the range(len(meta_strategy))
            all_pure_strategies_of_curr_profile = itertools.product(*[list(range(len(profile))) for p, profile in enumerate(current_profile)])

            # Then, for each possible pure strategy, get the probability 
            profile_utility_error = 0 
            profile_deviation_coverage_metric = 0
            deviations_accounted_for = 0
            for pure_strategy in all_pure_strategies_of_curr_profile:
                probability_of_pure_strategy = np.prod([current_profile[player][s] for player, s in enumerate(pure_strategy)])

                # CALCULATING UTILITY ERROR OF PROFILE
                profile_utility_error += all_utility_errors[pure_strategy] * probability_of_pure_strategy

            # CALCULATING UTILITY ERRORS OF DEVIATING STRATEGIES
            # Repeat for all players 
            for deviating_player in range(num_players):
                num_possible_deviations = num_strategies_per_player[deviating_player]

                for deviating_strategy in range(num_possible_deviations):
                    # Calculate the expected amount of utility error given everyone else plays according to the current profile 
                    # In essence, let's iterate through all possible pure strategies
                    # Only account for the ones with the corresponding deviation strategy
                    # Weight each contribution by the probability that all other players play such strategy
                    curr_dev_strategy_expected_error = 0
                    print("Calculating: dev player {} and dev strategy {}".format(deviating_player, deviating_strategy))

                    all_pure_strategies_in_full_game = itertools.product(*[list(range(num_strategies)) for num_strategies in num_strategies_per_player])
                    for pure_strategy in all_pure_strategies_in_full_game:
                        # print("Pure: ", pure_strategy, "  deviating_player: ", deviating_player, "  deviating_strategy: ", deviating_strategy)
                        if pure_strategy[deviating_player] != deviating_strategy:
                            continue 

                        # print([len(current_profile[p]) for p in range(num_players) if p != deviating_player])
                        if not all([pure_strategy[p] < len(current_profile[p]) for p in range(num_players) if p != deviating_player]):
                            continue

                        probability_of_other_players = np.prod([current_profile[p][pure_strategy[p]] for p in range(FLAGS.num_players) if p != deviating_player])
                        # print("in: ", pure_strategy, probability_of_other_players, all_utility_errors[pure_strategy])
                        curr_dev_strategy_expected_error += all_utility_errors[pure_strategy] * probability_of_other_players

                    print("Calculated: ", curr_dev_strategy_expected_error)
                    profile_deviation_coverage_metric += curr_dev_strategy_expected_error
                    deviations_accounted_for += 1

            profile_deviation_coverage_metric /= deviations_accounted_for

            all_profile_utility_errors.append(profile_utility_error)
            all_deviation_coverage_metrics.append(profile_deviation_coverage_metric)
        
        if not os.path.exists(FLAGS.save_graph_path):
            os.makedirs(FLAGS.save_graph_path)
        
        plt.plot(range(len(all_profile_utility_errors)), all_profile_utility_errors)
        plt.title("Utility Errors of Intermediate Final Profiles")
        plt.savefig(FLAGS.save_graph_path + "utility_errors_over_iterations.jpg")
        plt.clf()

        plt.plot(range(len(all_deviation_coverage_metrics)), all_deviation_coverage_metrics)
        plt.title("Deviation Coverage Metric of Intermediate Final Profiles")
        plt.savefig(FLAGS.save_graph_path + "deviation_coverage_metric_over_iterations.jpg")
        plt.clf()

        print("Utility Errors: ", all_profile_utility_errors)
        print("Deviation Coverages: ", all_deviation_coverage_metrics)


    # def extract_final_solution(self, pyspiel_game, trial_data_path, profile_files, solution="nash"):
    #     with open(trial_data_path + "model_meta_game.pkl", 'rb') as f:
    #         model_meta_game = pickle.load(f)

    #     # Construct the empirical game as desired by projectedReplicatorDynamics 

    #     ###################### Matrix Representation Reconstruction ###################################
    #     meta_game_shape = [max([pure_strategy[p] + 1 for pure_strategy in model_meta_game]) for p in range(FLAGS.num_players+1)]
    #     meta_games = []

    #     # Populate the metagame matrix with Nan values
    #     for _ in range(FLAGS.num_players+1):
    #         curr_player_meta_game = np.empty(meta_game_shape)
    #         curr_player_meta_game[:] = np.nan
    #         meta_games.append(curr_player_meta_game)
    #     #################### Matrix Representation Reconstruction End #################################
        
    #     #################### Populating Meta Game with Utilities #################################
    #     for pure_strategy, utilities_distribution in model_meta_game.items():
    #         expected_utilities = utilities_distribution["fixed"].expected_value
    #         for p in range(FLAGS.num_players+1):
    #             meta_games[p][pure_strategy] = expected_utilities[p]     
    #     ################## Populating Meta Game with Utilities End ###############################
    #     num_solutions = meta_game_shape[0] - 1
    #     print("With meta-game shape of {}, we assume that there are {} solutions to extract.".format(meta_game_shape, num_solutions))
    #     for j in range(num_solutions):

    #         # Extract relevant subgame
    #         # If statement takes care of env player. The else statement is for actual players
    #         # indices = [slice(0, meta_game_shape[p], 1) if p == FLAGS.num_players else slice(0, j+1) for p in range(FLAGS.num_players+1)]

    #         # NOTE: only valid for two player games
    #         subgames = [meta_games[p][0:j+2, 0:j+2, 0:meta_game_shape[2]] for p in range(FLAGS.num_players+1)]

    #         print("Loading profile from: ", trial_data_path+profile_files[j])
    #         with open(trial_data_path+profile_files[j], 'rb') as f:
    #             initial_profiles = pickle.load(f)

    #         if len(initial_profiles) == 2:
    #             env_profile = [1.0 / meta_game_shape[2]] * meta_game_shape[2]
    #             initial_profiles.append(env_profile)

    #         if solution.lower() == "nash":
    #             result = projected_replicator_dynamics.regularized_replicator_dynamics(
    #                 subgames, regret_lambda=0, symmetric=FLAGS.symmetric, prd_initial_strategies=initial_profiles, symmetric_players=[[i for i in range(FLAGS.num_players)]], unused_kwargs={}) 
    #         else:
    #             raise NotImplementedError
            
    #         if len(result) > FLAGS.num_players:
    #             result = result[:FLAGS.num_players]

    #         with open(trial_data_path+"final_solution_iteration_{}.pkl".format(j), 'wb') as f:
    #             pickle.dump(result, f)
    #         print("Using subgame: ", subgames)
    #         print("Saved new profile at ", trial_data_path+"final_solution_iteration_{}.pkl".format(j), ": ", result)
    #         print("Previous profile: ", initial_profiles[0])
    #         print("\n")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print("\n\n")
    np.random.seed(FLAGS.seed)

    # Get the game-specific true state extractor 
    if FLAGS.game_name == "bargaining":
        pyspiel_game = pyspiel.load_game(FLAGS.game_name, {"discount": 0.99 , "instances_file":"../../../../games/bargaining_instances_symmetric_25pool_25valuations.txt"})
        true_state_extractor = BargainingTrueStateGenerator(FLAGS.game_name, pyspiel_game.information_state_tensor_shape()[0])
    if FLAGS.game_name == "bargaining_generalized":
        pyspiel_game = pyspiel.load_game(FLAGS.game_name, {"discount": 0.99})
        true_state_extractor = BargainingGeneralizedTrueStateGenerator(FLAGS.game_name, pyspiel_game.information_state_tensor_shape()[0])
    if FLAGS.game_name == "leduc_poker":
        true_state_extractor = LeducPokerTrueStateGenerator(FLAGS.game_name)
        pyspiel_game = pyspiel.load_game(FLAGS.game_name)
    
    logging.info("Loaded game: %s", FLAGS.game_name)
    true_state_extractor.get_set_info_depending_on_game(pyspiel_game)

    # Use rl_environment as a wrapper and specify the observation type
    env = rl_environment.Environment(pyspiel_game, observation_type=rl_environment.ObservationType.INFORMATION_STATE)

    evaluation_module = TrueGameRegretEvaluationModule(env, FLAGS.num_players, FLAGS.evaluation_strategy_path, FLAGS.save_path, FLAGS.symmetric, FLAGS.num_simulations, true_state_extractor)

    response_parameters = {
        "state_representation_size": env._game.information_state_tensor_shape()[0], 
        "num_actions": env.action_spec()["num_actions"],
        "double": True, 
        "hidden_layers_sizes": [200] * 2, 
        "replay_buffer_capacity": int(5e4), 
        "batch_size": 64, 
        "learning_rate": 1e-4,
        "update_target_network_every": 1000, 
        "learn_every": 2, 
        "discount_factor": .99, 
        "min_buffer_size_to_learn": int(2e4), 
        "epsilon_start": 1.0, 
        "epsilon_end": .02, 
        "epsilon_decay_duration": int(2e5),
    }

    final = FLAGS.eval_regret_with_final_profiles
    all_profile_files = [f for f in os.listdir(FLAGS.save_path) if (("profile" in f) if not final else "solution" in f) and (".pkl" in f) and ("final" in f if final else "final" not in f)]
    all_profile_files = sorted(all_profile_files, key=lambda s: int(s.split('_')[-1].split('.')[0]))

    if FLAGS.evaluation_name == "add_evaluation_strategy":
        for profile_file_name in all_profile_files[::]:
            start = time.time()
            evaluation_module.train_true_game_best_response(FLAGS.save_path, ["player_0/"], profile_file_name, training_player=0, training_steps_total=int(2e5), response_parameters=response_parameters)
            print("Finished training true game best response to {} in {} seconds.".format(profile_file_name, time.time() - start))
    elif FLAGS.evaluation_name == "evaluate_regret":
        evaluation_module.evaluate_regret(FLAGS.save_path, ["player_0/"], final=FLAGS.eval_regret_with_final_profiles)
    elif FLAGS.evaluation_name == "evaluate_utility_error":
        evaluation_module.evaluate_trajectory_level_reward_uncertainty_correlations(pyspiel_game, FLAGS.save_path, ["player_0/"], true_state_extractor, FLAGS.save_graph_path, response_parameters)
    # elif FLAGS.evaluation_name == "extract_final_solution":
    #     evaluation_module.extract_final_solution(pyspiel_game, FLAGS.save_path)
    elif FLAGS.evaluation_name == "full_evaluation_pipeline":
        # Extract final solution for each PSRO iteration 
        # Get the in-algorithm solutions
        # all_profile_files = [f for f in os.listdir(FLAGS.save_path) if (("profile" in f)) and (".pkl" in f) and ("final" not in f)]
        # all_profile_files = sorted(all_profile_files, key=lambda s: int(s.split('_')[-1].split('.')[0]))
        # evaluation_module.extract_final_solution(pyspiel_game, FLAGS.save_path, all_profile_files)

        # Get the final solutions
        all_profile_files = [f for f in os.listdir(FLAGS.save_path) if ("solution" in f) and (".pkl" in f) and ("final" in f)]
        all_profile_files = sorted(all_profile_files, key=lambda s: int(s.split('_')[-1].split('.')[0]))

        # Train true game best responses by adding evaluation strategy 
        for profile_file_name in all_profile_files[::]:
            start = time.time()
            evaluation_module.train_true_game_best_response(FLAGS.save_path, ["player_0/"], profile_file_name, training_player=0, training_steps_total=int(2e5), response_parameters=response_parameters)
            print("Finished training true game best response to {} in {} seconds.".format(profile_file_name, time.time() - start))
        
        # Evaluate regret 
        evaluation_module.evaluate_regret(FLAGS.save_path, ["player_0/"], final=True)
    elif FLAGS.evaluation_name == "analyze_deviation_coverage":
        # Get the final solutions
        all_profile_files = [f for f in os.listdir(FLAGS.save_path) if ("solution" in f) and (".pkl" in f) and ("final" in f)]
        all_profile_files = sorted(all_profile_files, key=lambda s: int(s.split('_')[-1].split('.')[0]))

        evaluation_module.analyze_deviation_coverage_by_iteration(FLAGS.num_players, FLAGS.save_path, all_profile_files)

    print("Experiment done.")

if __name__ == "__main__":
  app.run(main)
