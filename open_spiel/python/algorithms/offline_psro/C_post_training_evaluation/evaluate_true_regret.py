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
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.ddqn_simple import DQN
from open_spiel.python.algorithms.offline_psro.B_training.main_components.offline_psro_wrapper import calculate_expected_regret, get_curr_and_deviation_payoffs
from open_spiel.python.algorithms.offline_psro.B_training.tf_model_management.tf_model_management import TFModelManagement
from open_spiel.python.rl_environment import StepType
from open_spiel.python.algorithms.offline_psro.utils.utils import generate_single_rollout, Step, get_graphs_and_context_managers
from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.bargaining_true_state_generator import BargainingTrueStateGenerator
from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.leduc_poker_true_state_generator import LeducPokerTrueStateGenerator
from open_spiel.python import rl_environment

from open_spiel.python.algorithms import projected_replicator_dynamics

FLAGS = flags.FLAGS 

# Simulation related
flags.DEFINE_bool("symmetric", False, "Is the game symmetric?")
flags.DEFINE_integer("num_players", 2, "Number of players in the game")
flags.DEFINE_string("save_path", "random_runs/", "Folder to save all models and data")
flags.DEFINE_integer('num_simulations', 1000, "Number of simulations to estimate normal-form game entries")
flags.DEFINE_integer("num_bootstrapped_samples", 1000, "Number of bootstrapped samples when estimating expected regret")
flags.DEFINE_string("evaluation_strategy_path", "", "Folder to save or load any evaluation set strategies")
flags.DEFINE_integer("training_steps", int(2e5), "Number of training steps for true game strategies")
flags.DEFINE_string("game_name", "", "Name of game")
flags.DEFINE_bool("eval_regret_with_final_profiles", False, "Whether to use the extracted final profiles at each iteration to calculate regret, not the MSS used during the trial")

flags.DEFINE_string("evaluation_name", "evaluate_regret", "Whether to train a new evaluation strategy or evaluate reget of a trial")

# Misc
flags.DEFINE_integer("seed", 1, "Seed for random")


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

    def train_true_game_best_response(self, trial_data_path, player_trial_strategy_path, profile_file_name, training_player, training_steps, response_parameters):
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

            # Create a new computational graph for our new policy
            new_rl_training_session = stack.enter_context(tf.Session(graph=tf.Graph()))
            sessions[training_player].append(new_rl_training_session)

            # Initialize new policy and append
            new_policy = DQN(new_rl_training_session, graph=new_rl_training_session.graph, **response_parameters)

            # If symmetric, then all players share policies
            if FLAGS.symmetric and len(all_policies) == 1:
                all_policies = [all_policies[0] for _ in range(FLAGS.num_players)]
                sessions = [sessions[0] for _ in range(FLAGS.num_players)]

            # Training loop. Use an rl_environment and manual step creations to provide learning to the currently training response
            aggregate_rewards = []
            num_training_steps = 0
            while num_training_steps < FLAGS.training_steps:
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
                    curr_step = Step(info_state=timestep.observations["info_state"][curr_player], reward=None, is_terminal=False, halted=False,
                                    legal_actions_mask=[1 if a in timestep.observations["legal_actions"][curr_player] else 0 for a in range(self._env.action_spec()["num_actions"])],
                                    acting_players=[curr_player], global_state=None)
                
                done = False
                returns = 0
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
                            curr_step = Step(info_state=timestep.observations["info_state"][curr_player], reward=[timestep.rewards], is_terminal=done, halted=False,
                                            legal_actions_mask=[1 if a in timestep.observations["legal_actions"][curr_player] else 0 for a in range(self._env.action_spec()["num_actions"]) ],
                                            acting_players=[curr_player], global_state=None)
                    else:
                        raise NotImplementedError
                
                # Last time step to account for terminal state
                for p in range(self._num_players):
                    curr_step = Step(info_state=timestep.observations["info_state"][p], reward=[timestep.rewards], is_terminal=done, halted=False,
                                            legal_actions_mask=[1 if a in timestep.observations["legal_actions"][p] else 0 for a in range(self._env.action_spec()["num_actions"]) ],
                                            acting_players=[p], global_state=None)
                    policies[p].step(curr_step, p, curr_sessions[p], is_evaluation=False)  
                aggregate_rewards.append(returns)

            # print("Aggregate rewards for trained policy: ", aggregate_rewards)
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
            num_strategies = len([f for f in os.listdir(trial_data_path + player_path) if ("policy" in f) and (".pb" in f)])
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

        regret_values = [[] for _ in range(self._num_players)]

        for profile in all_profiles:
            # This is a minute but important point: we take [1:] of the player profile because we don't save the initial, singleton policy in PSRO
            profile = [np.pad(player_profile[1:], (0, num_strategies_each_player[player] - player_profile[1:].shape[0])) for player, player_profile in enumerate(profile)]

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

    def extract_final_solution(self, trial_data_path, solution_concept, symmetric):
        """
        Given the meta-game that was found during a PSRO run, extract the final profile that would be outputted as if 
        the PSRO were to terminate at that iteration. Then, save the profile. This is done so that we appropriately evaluate
        strategy exploration and stay consistent in how we evaluate the quality of strategy sets across iterations.

        trial_data_path: relative path containing data of the meta-games we want to extract our final solution out of
        solution_concept: dictionary describing the operation we apply on our meta-game to extract a profile 
        symmetric: symmetric game(?) and profile(?)

        solution_concept fields:
        evaluation_halt_type: string specifying which HALT distribution to use when evaluating each entry in the meta-game
        evaluation_criteria: string describing how we determine the quality of a particular profile 
        meta_strategy_solver: string describing the meta-strategy solver to be used after the meta-game is created
        num_iterations: integer determining how many times we run the meta_strategy_solver before extracting profile with best evaluation_criteria score
        """

        meta_game_files = [f for f in os.listdir(trial_data_path) if "meta_game" in f]
        meta_game_files = sorted(meta_game_files, key=lambda s: int([token for token in (s.split('.')[0]).split('_') if token.isdigit()][0]))
        
        for iteration, meta_game_file in enumerate(meta_game_files):
            total_path = trial_data_path + meta_game_file
            with open(total_path, 'rb') as f:
                meta_game = pickle.load(f)
            print("Extracting solution from profile at iteration {} file name {}.".format(iteration, meta_game_file))

            # Reconstruct the meta game using evaluation_halt_type distribution
            num_players = len(list(meta_game.keys())[0])
            empirical_game_shape = [0 for _ in range(num_players)]
            for strategy in meta_game.keys():
                for p in range(num_players):
                    empirical_game_shape[p] = max(empirical_game_shape[p], strategy[p] + 1)
            
            meta_game_matrix = np.empty(empirical_game_shape + [num_players])
            meta_game_matrix[:] = np.nan 

            for strategy, utility_dictionary in meta_game.items():
                if solution_concept["evaluation_halt_type"] == "fixed":
                    meta_game_matrix[strategy] = utility_dictionary["fixed"].expected_value
                elif solution_concept["evaluation_halt_type"] == "optimistic":
                    meta_game_matrix[strategy] = utility_dictionary["fixed"].expected_value
                else:
                    raise NotImplementedError

            meta_game_matrix_transposed = np.transpose(meta_game_matrix, [len(np.array(meta_game_matrix).shape) - 1] + list(range(0, len(np.array(meta_game_matrix).shape) - 1)))

            if solution_concept["evaluation_criteria"] == "expected_regret":
                def expected_regret(profile):
                    average_sumRegret = 0
                    for _ in range(solution_concept["num_bootstrapped_samples"]):
                        sumRegret = 0
                        optimistic_meta_game_matrix = np.empty(empirical_game_shape + [num_players])
                        optimistic_meta_game_matrix[:] = np.nan

                        for strategy, utility_dictionary in meta_game.items():
                            if solution_concept["evaluation_halt_type"] == "fixed":
                                optimistic_meta_game_matrix[strategy] = utility_dictionary["fixed"].expected_value
                            elif solution_concept["evaluation_halt_type"] == "optimistic":
                                if all([s == 0 for s in strategy]):
                                    optimistic_meta_game_matrix[strategy] = utility_dictionary["fixed"].expected_value
                                else:
                                    optimistic_meta_game_matrix[strategy] = utility_dictionary["optimistic"].sample()

                        for p in range(num_players):
                            # deviation_payoffs, curr_payoff = get_curr_and_deviation_payoffs(meta_game.keys(), meta_game_matrix, profile, p)
                            deviation_payoffs, curr_payoff = get_curr_and_deviation_payoffs(meta_game.keys(), optimistic_meta_game_matrix, profile, p)
                            sumRegret += max(max(deviation_payoffs) - curr_payoff, 0)
                        average_sumRegret += sumRegret
                    average_sumRegret = average_sumRegret / solution_concept["num_bootstrapped_samples"]
                    return average_sumRegret
                evaluation_criteria = expected_regret
            elif solution_concept["evaluation_criteria"] == "deviation_coverage":
                def deviation_coverage(profile):
                    
                    non_halt_proportion_meta_game = np.empty(empirical_game_shape + [num_players])
                    non_halt_proportion_meta_game[:] = np.nan 
                    for strategy, utility_dictionary in meta_game.items():
                        non_halt_proportion_meta_game[strategy] = np.array([utility_dictionary["non_halt_proportion"] for _ in range(num_players)])
                    
                    profile_coverage, average_deviation_coverage = 0, 0
                    for p in range(num_players):
                        deviation_coverage, curr_coverage = get_curr_and_deviation_payoffs(meta_game.keys(), non_halt_proportion_meta_game, profile, p)
                        average_deviation_coverage += np.mean(deviation_coverage)
                        profile_coverage += curr_coverage
                    
                    profile_coverage = profile_coverage / num_players
                    average_deviation_coverage = average_deviation_coverage / num_players
                    return solution_concept["weight_deviation_coverage"] * average_deviation_coverage + (1 - solution_concept["weight_deviation_coverage"]) * profile_coverage 

                evaluation_criteria = deviation_coverage
            else:
                raise NotImplementedError

            best_profile, best_criteria = None, -np.inf
            for _ in range(solution_concept["num_iterations"]):
                # Apply the MSS specificed by meta_strategy_solver
                if self._symmetric:
                    initial_strategies = [np.random.choice(range(1, 11)) for _ in range(meta_game_matrix[0].shape[0])]
                    initial_strategies = [initial_strategies for _ in range(num_players)]
                else:
                    initial_strategies = [[np.random.choice(range(1,11)) for _ in range(empirical_game_shape[k])] for k in range(num_players)]
                initial_strategies = [np.array(mixture) / np.sum(mixture) for mixture in initial_strategies]

                if solution_concept["meta_strategy_solver"] == 'rd':
                    result = projected_replicator_dynamics.regularized_replicator_dynamics(
                        meta_game_matrix_transposed, prd_initial_strategies=initial_strategies, regret_lambda=1e-4, symmetric=symmetric, unused_kwargs={}, prd_iterations=int(1e5)) 
                else:
                    raise NotImplementedError

                criteria_value = evaluation_criteria(result)
                
                if criteria_value > best_criteria:
                    best_profile = result
                    best_criteria = criteria_value

            print("Found best profile {} with criteria value {}.".format(best_profile, best_criteria))
            
            with open(trial_data_path+"final_solution_iteration_{}.pkl".format(iteration), 'wb') as f:
                pickle.dump(best_profile, f)
            
            print("Saved best profile to {}".format(trial_data_path+"final_solution_iteration_{}.pkl".format(iteration)))

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print("\n\n")
    np.random.seed(FLAGS.seed)

    # Get the game-specific true state extractor 
    if FLAGS.game_name == "bargaining":
        pyspiel_game = pyspiel.load_game(FLAGS.game_name, {"discount": 0.99})
        true_state_extractor = BargainingTrueStateGenerator(FLAGS.game_name, pyspiel_game.information_state_tensor_shape()[0])
        max_episode_length = pyspiel_game.max_game_length()
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
        "hidden_layers_sizes": [50] * 2, 
        "replay_buffer_capacity": int(2e4), 
        "batch_size": 64, 
        "learning_rate": 3e-5,
        "update_target_network_every": 1000, 
        "learn_every": 5, 
        "discount_factor": .99, 
        "min_buffer_size_to_learn": 1000, 
        "epsilon_start": .8, 
        "epsilon_end": .02, 
        "epsilon_decay_duration": int(1e5) 
    }

    final = FLAGS.eval_regret_with_final_profiles
    all_profile_files = [f for f in os.listdir(FLAGS.save_path) if (("profile" in f) if not final else "solution" in f) and (".pkl" in f) and ("final" in f if final else "final" not in f)]
    all_profile_files = sorted(all_profile_files, key=lambda s: int(s.split('_')[-1].split('.')[0]))

    if FLAGS.evaluation_name == "add_evaluation_strategy":
        for profile_file_name in all_profile_files[::]:
            start = time.time()
            evaluation_module.train_true_game_best_response(FLAGS.save_path, ["player_0/"], profile_file_name, training_player=0, training_steps=FLAGS.training_steps, response_parameters=response_parameters)
            print("Finished training true game best response to {} in {} seconds.".format(profile_file_name, time.time() - start))
    elif FLAGS.evaluation_name == "evaluate_regret":
        evaluation_module.evaluate_regret(FLAGS.save_path, ["player_0/"], final=FLAGS.eval_regret_with_final_profiles)
    elif FLAGS.evaluation_name == "get_final_profiles":
        solution_concept = {"evaluation_halt_type": "optimistic",
                            "evaluation_criteria": "deviation_coverage",
                            "meta_strategy_solver": "rd",
                            "num_iterations": 200, 
                            "num_bootstrapped_samples": FLAGS.num_bootstrapped_samples,
                            "weight_deviation_coverage": .5}
        evaluation_module.extract_final_solution(FLAGS.save_path, solution_concept, FLAGS.symmetric)
    print("Experiment done.")

if __name__ == "__main__":
  app.run(main)
