"""
Main wrapper for offline PSRO.
"""


import time
import collections
import numpy as np
import os
import pyspiel
import tensorflow.compat.v1 as tf
import sys
import itertools 
import pickle

from contextlib import ExitStack 
from datetime import datetime
from absl import app
from absl import flags
import matplotlib.pyplot as plt
from absl import logging

from open_spiel.python import rl_environment
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.world_model_deterministic import WorldModelDeterministic
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.incentive_shaper import IncentiveShaper
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.ddqn_simple import DQN
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.uniform_random_policy import UniformRandomPolicy
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.behavior_cloning import JointBehaviorCloning

from open_spiel.python.algorithms.offline_psro.utils.utils import generate_single_rollout, compute_hash_string
from open_spiel.python.algorithms.offline_psro.utils.utils import Step

from open_spiel.python.algorithms import projected_replicator_dynamics

############ Utility Distribution Class for Ease of Representation #############
class UtilityDistribution:
    def __init__(self, data):
        self._data = data # [N, num_players]
    
    @property
    def expected_value(self):
        return np.mean(self._data, axis=0)

    def sample(self):
        return self._data[np.random.choice(self._data.shape[0])]
########## Utility Distribution Class for Ease of Representation End ###########

########## Profile Evaluation Stuff #############
def calculate_expected_regret(profile, empirical_game, expected_regret_estimation_iterations, players):
    # Given a joint profile (length self._num_players) and a dictionary representation of our empirical game, find the expected regret through bootstrap

    sumRegret_samples = []
    for _ in range(expected_regret_estimation_iterations):
        # Reconstruct the empirical game in meta_game form by sampling from each empirical game entry
        curr_empirical_game = np.empty([len(player_profile) for player_profile in profile] + [players])
        curr_empirical_game[:] = np.nan 

        for strategy, utility in empirical_game.items():
            curr_empirical_game[strategy] = utility.sample()

        # Then, calculate regret for each of the players (sum regret)
        sumRegret = 0
        for p in range(players):
            deviation_payoffs, curr_payoff = get_curr_and_deviation_payoffs(empirical_game.keys(), curr_empirical_game, profile, p)
            
            player_regret = np.max(deviation_payoffs) - curr_payoff 
            sumRegret += player_regret
        
        sumRegret_samples.append(sumRegret)

    # Average over all the various regret values we found
    return np.mean(sumRegret_samples)

def get_curr_and_deviation_payoffs(all_pure_strategies, empirical_game, profile, player):
    player_profile = profile[player]

    relevant_empirical_game = empirical_game[..., player]
    deviation_payoffs = np.zeros(len(profile[player]))
    curr_payoff = 0
    for strategy in all_pure_strategies:
        utility = relevant_empirical_game[strategy]
        prob_others = np.prod([curr_profile[strategy[i]] for i, curr_profile in enumerate(profile) if i != player])
        prob_all = prob_others * player_profile[strategy[player]]

        deviation_payoffs[strategy[player]] += prob_others * utility 
        curr_payoff += prob_all * utility
    return deviation_payoffs, curr_payoff
######## Profile Evaluation Stuff End ###########

######## Test Stuff ##########

IncentiveDatapoint = collections.namedtuple(
    "IncentiveDatapoint",
    "aggregate_hash global_state action player")

####### End Test Stuff #######

class OfflineModelBasedPSRO:

    def __init__(self, dataset, symmetric, is_turn_based, num_players, num_actions, initial_strategies, model_args, tf_model_management_module, save_path, true_state_extractor, max_episode_length, num_iterations_distribution, incentive_alpha, verbose=True):
        """
            dataset: a list of trajectories, each of which are a list of Transition namedTuples (utils)
            symmetric: indicates whether the game is symmetric for simplicity
            is_turn_based: indicates whether the game is turn_based or simultaneous
            num_players: how many players in the game
            initial_strategies: list of strategies to initialize the normal form game with. ideally, these are the behavior policies but can be anything
            model_args: dictionary describing parameters for training the ensemble-based model
            tf_model_management_module: TFModelManagement object that handles model saving, loading, and freezing
            save_path: string specifying the relative path to folder in which we will be saving all models, policies, and data relevant to this PSRO run
            true_state_extractor: module that provides manual, game-specific information for info state generation and observation handling 
            max_episode_length: max number of steps to rollout the model
        """
        ################################## Initializations ######################################
        self._data = dataset
        self._symmetric = symmetric
        self._num_players = num_players
        self._num_actions = num_actions
        self._tf_model_management = tf_model_management_module
        self._true_state_extractor = true_state_extractor
        self._is_turn_based = is_turn_based
        self._max_episode_length = max_episode_length
        self._verbose = verbose

        self._incentive_alpha = incentive_alpha
        assert self._incentive_alpha < 1.0

        # self._iterations_with_optimistic_strategy = []
        self._empirical_game = None 
        self._num_iterations_distribution = num_iterations_distribution
        self._optimistic_iteration = False
        self._optimistic_distribution_initialized = False
        self._save_path = save_path if save_path[-1] == "/" else save_path + "/"

        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)

        ################################ Initializations End ####################################
        
        ################################ Model Training #########################################
        with tf.Session() as sess:
        # Check what kind of model we want to train (deterministic or stochastic world model)
            model_args["session"] = sess
            if model_args["model_type"] == "deterministic":
                self._model = WorldModelDeterministic(model_args["state_size"], model_args["action_size"], model_args) # Simple MLP
            else:
                raise NotImplementedError # World model MDN
            
            # Train and Losses
            losses = self._model.train(num_gradient_steps=model_args["training_steps"])

            # Save the model
            tf_model_management_module.save(sess, self._save_path + "model.ckpt")
        
        # Plot generation and saving
        plt.plot(range(len(losses)), losses)
        plt.savefig(self._save_path+'model_loss.jpg')
        plt.clf()
        
        # Outside of the session, load and freeze the model.
        self._model.freeze(tf_model_management_module, self._save_path)
        ############################## Model Training End #######################################

        
        #################### Strategy and Profile Initializations ####################
        if self._symmetric:
            if len(initial_strategies) > 1:
                logging.error("{} initial strategies were passed in for symmetric game.".format(len(initial_strategies)))
                raise Exception 
            if len(initial_strategies) == 1:
                self._strategy_set = initial_strategies 
                self._profile = [1.0]
            else: 
                self._strategy_set = initial_strategies
                self._profile = []
                logging.info(" No initial strategies were provided. Ensure to train an initial strategy with incentive alpha set to 1.0 to insert an initial strategy optimizing for deviation coverage.")
        else:
            if len(initial_strategies) != self._num_players:
                logging.error("Number of strategies in initial_strategies {} does not match the number of players {}".format(len(initial_strategies), self._num_players))
                raise Exception 
            if len(initial_strategies) == self._num_players:
                self._strategy_set = [[initial_strategies[i]] for i in range(self._num_players)]
                self._profile = [[1.0] for i in range(self._num_players)]
                
            elif len(initial_strategies) == 0:
                self._strategy_set = [[] for i in range(self._num_players)]
                self._profile = [[] for i in range(self._num_players)]
                logging.info(" No initial strategies were provided. Ensure to train an initial strategy with incentive alpha set to 1.0 to insert an initial strategy optimizing for deviation coverage.")
        ################## Strategy and Profile Initializations End ##################

    # def train_coverage_incentive(self, global_state_size, num_actions, halt_threshold):
    #     # Initialize uniform random action policies (purely random no consideration of in-distribution of out-of-distribution actions)
    #     uniform_policies = [UniformRandomPolicy(state_size=1, num_actions=num_actions) for _ in range(self._num_players)]

    #     # Rollout the uniform random action policies a bunch of times 
    #     training_steps = int(3e4)
    #     episode_batch = 10
    #     losses = []
    #     print("Training incentive reward. ")
    #     map_hash_to_training_info = {}
    #     with ExitStack() as stack:
    #         model_session = stack.enter_context(tf.Session(graph=self._model.get_frozen_graph()))

    #         sess = stack.enter_context(tf.Session(graph=tf.Graph()))
    #         self._incentive_model = IncentiveShaper(global_state_size, num_actions, self._num_players, session=sess)
    #         for _ in range(training_steps):
    #             # Load model graph 
    #             state_batch = []
    #             action_batch = []
    #             reward_batch = []
    #             player_batch = []

    #             for _ in range(episode_batch):
    #                 _, _, halt, rollout, actions = self.rollout_model(uniform_policies, policy_sessions=[None for _ in range(self._num_players)], model_session=model_session, halt_threshold=halt_threshold, is_evaluation=True, kappa=0, optimistic_player=0)
                    
    #                 for t, step in enumerate(rollout[:-1]):
    #                     if self._is_turn_based:
    #                         state_batch.append(step.global_state)
    #                         action_batch.append(actions[t][0])
    #                         reward_batch.append([0 if halt else 1])
    #                         player_batch.append([step.acting_players[0]])

    #                         aggregate_hash = compute_hash_string(step.global_state) + compute_hash_string(actions[t]) + compute_hash_string(step.acting_players)
    #                         datapoint = IncentiveDatapoint(aggregate_hash=aggregate_hash, global_state=step.global_state, action=actions[t][0], player=step.acting_players[0])
    #                         if aggregate_hash in map_hash_to_training_info:
    #                             map_hash_to_training_info[aggregate_hash][1].append(0 if halt else 1)
    #                         else:
    #                             map_hash_to_training_info[aggregate_hash] = [datapoint, [0 if halt else 1]]
    #                     else:
    #                         for p in step.acting_players:
    #                             state_batch.append(step.global_state)
    #                             action_batch.append([actions[t][p]])
    #                             reward_batch.append([0 if halt else 1])
    #                             player_batch.append([p])
    #                 # Use these as training targets using supervised learning

    #             loss = self._incentive_model.one_step_learn(state_batch, action_batch, reward_batch, player_batch)
    #             losses.append(loss)
            
    #         window = 50
    #         average_losses = [sum(losses[i:i+window]) / window for i in range(len(losses) - window + 1)]

    #         self._tf_model_management.save(sess, self._save_path + "incentive_model.ckpt")

    #         plt.plot(range(len(average_losses)), average_losses)
    #         plt.savefig(self._save_path + "incentive_model_loss.jpg")
    #         plt.clf() 

    #     self._incentive_model.freeze(self._tf_model_management, self._save_path)

    #     for hash_aggregate, datapoint_and_reward in map_hash_to_training_info.items():
    #         datapoint_and_reward[1] = np.mean(datapoint_and_reward[1])
        
    #     with ExitStack() as stack:
    #         frozen_session = stack.enter_context(tf.Session(graph=self._incentive_model.get_frozen_graph()))

    #         all_diffs = []
    #         for hash_aggregate, datapoint_and_mean_reward in map_hash_to_training_info.items():
    #             s = datapoint_and_mean_reward[0].global_state 
    #             a = datapoint_and_mean_reward[0].action
    #             p = datapoint_and_mean_reward[0].player 
    #             mean_reward = datapoint_and_mean_reward[1]
    #             predicted_reward = self._incentive_model.predict(s, a, p, frozen_session)
    #             diff = np.square(mean_reward - predicted_reward)
    #             all_diffs.append(diff)
    #         print("Average square diff: ", np.mean(all_diffs))
    #     # print(yes)

    def set_incentive_alpha(self, alpha):
        self._incentive_alpha = alpha
        logging.info("Incentive alpha set to {}".format(self._incentive_alpha))
    
    def train_and_add_strategy(self, response_parameters, training_player, num_training_steps, halt_reward_type, kappa, halt_threshold):
        """
            response_parameters: describes the type of RL algorithm and parameters for it that will be used to train new algorithm
            training_player: player index whose strategy set we are adding to
            num_training_steps: number of environment interactions the training_player will receive during training
            halt_reward_type: if there is a HALTED trajectory under our model, what kind of terminal reward will be apply?
            kappa: MoREL-style penalty for reaching a HALT state
        """
        ############################## Initializations ##############################
        # If symmetric, then strategy set and profile are associated with only one of the players. Repeat for each player for coding simplicity
        if self._symmetric:
            self._strategy_set = [self._strategy_set for _ in range(self._num_players)]
            self._profile = [self._profile for _ in range(self._num_players)]

        # Get computational graphs for each of the frozen policies/models and load them into our context manager
        all_graphs = [[pol.get_graph() for pol in player_set] for player_set in self._strategy_set]
        context_managers = [[tf.Session(graph=g) for g in player_set] for player_set in all_graphs]
        optimistic_training = False # self._optimistic_iteration and (halt_reward_type == "optimistic")
        logging.info("Training new strategy. Optimistic: {}    Halt Type: {}".format(optimistic_training, "optimistic" if optimistic_training else "fixed"))
        # self._iterations_with_optimistic_strategy.append(len(self._strategy_set[0])) # strategy index of new strategy (to be optimistic)
        ############################ Initializations End ############################

        ############################ Training Loop ##################################
        # Load all the sessions and graphs from the previously frozen policies
        with ExitStack() as stack:
            # Load policy graphs 
            sessions = [[stack.enter_context(cm) if cm != None else None for cm in cm_list] for cm_list in context_managers]
            
            # Load model graph
            model_session = stack.enter_context(tf.Session(graph=self._model.get_frozen_graph()))

            # Incentive context 
            incentive_session = None # stack.enter_context(tf.Session(graph=self._incentive_model.get_frozen_graph()))            

            # Create a new computational graph for our new policy
            new_rl_training_session = stack.enter_context(tf.Session(graph=tf.Graph()))
            sessions[training_player].append(new_rl_training_session)


            # Initialize new policy and append
            new_policy = DQN(new_rl_training_session, graph=new_rl_training_session.graph, **response_parameters)
            # TODO: Extract the input and output variables from our saved variables and initialize in new_policy as well!
            self._strategy_set[training_player].append(new_policy)

            # Training loop for new policy
            curr_training_steps = 0
            aggregate_returns = []
            while curr_training_steps < num_training_steps:
                # Sample from profiles to get response target
                policies, policy_sessions = [], []
                optimize_deviation_coverage = np.random.random() < self._incentive_alpha
                for p in range(self._num_players):
                    if p == training_player:
                        policies.append(self._strategy_set[p][len(self._strategy_set[training_player]) - 1])
                        policy_sessions.append(sessions[p][len(self._strategy_set[training_player]) - 1])
                    else:
                        if optimize_deviation_coverage:
                            policies.append(UniformRandomPolicy(state_size=1, num_actions=self._num_actions))
                            policy_sessions.append(None)
                        else:
                            strategy_choice = np.random.choice(list(range(len(self._profile[p]))), p=self._profile[p])
                            policies.append(self._strategy_set[p][strategy_choice])
                            policy_sessions.append(sessions[p][strategy_choice])

                 # TODO: If this is trial where we are responding to a UniformRandomAction, call the method SetDeviationCoverage optimization in the training_player's policy. 
                policies[training_player].set_deviation_coverage_flag(optimize_deviation_coverage)
                
                # Do one rollout using the trained model. Set is_evaluation to False to train new_policy
                # halt_threshold = 1e9 if optimistic_training else halt_threshold
                returns, steps, _, _, _= self.rollout_model(policies, policy_sessions, model_session, halt_threshold, is_evaluation=False, kappa=kappa, optimistic_player=training_player, apply_incentive=True, incentive_session=incentive_session)

                aggregate_returns.append(returns[training_player])
                
                # Track the number of steps for the currently training player
                curr_training_steps += steps[training_player] 

                # Returns may be a distribution now 
                # aggregate_returns.append(np.sum(returns * np.expand_dims(np.reshape(probs, [-1]), axis=1), axis=0)[training_player])
            print("\nInformation on training player's policy.     Proportion trajectories relabeled: {}       Proportion trajectories non-halted of relabeled: {}\n".format(policies[training_player]._num_relabeled_trajectories / policies[training_player]._num_trajectories, 
                                                                                                                                                               policies[training_player]._num_relabeled_non_halted_trajectories / policies[training_player]._num_relabeled_trajectories))

            window = 200
            averaged_over = [sum(aggregate_returns[i:i+window]) / window for i in range(len(aggregate_returns) - window + 1)]
            plt.plot(list(range(len(averaged_over))), averaged_over)

            # To distinguish between different players, we create player-specific directories for their policies.
            # In symmetric games, there will only be a folder for player 0 (because strategies are shared)
            policy_save_path = self._save_path + "player_{}/".format(training_player)
            if not os.path.exists(policy_save_path):
                os.makedirs(policy_save_path)
                
            plt.savefig(policy_save_path+'policy_model_rewards_policy_{}_player_{}.jpg'.format(len(self._strategy_set[training_player]), training_player))
            plt.clf()
            
            # After training, save the currently training policy
            self._tf_model_management.save(new_rl_training_session, policy_save_path + 'policy_{}.ckpt'.format(new_policy._id))
    
        # Outside of the session, freeze and load the newly trained policy. 
        new_policy.freeze(self._tf_model_management, policy_save_path)
        ########################## Training Loop End ################################

        # If symmetric, we only keep track of one player's strategies and profile 
        if self._symmetric:
            self._strategy_set = self._strategy_set[0]
            self._profile = self._profile[0]

    def update_empirical_game(self, num_simulations, kappa=5, halt_threshold=1):
        """
        Update entire empirical game with utility distributions for each entry.
        
        num_simulations: number of model rollouts to use to best estimate utilities
        kappa: MoREL penalty for going out-of-distribution
        """
        # If symmetric, then strategy sets are for one player. For coding simplicity, make copies 
        if self._symmetric:
            self._strategy_set = [self._strategy_set for _ in range(self._num_players)]
        
        num_strategies = len(self._strategy_set[0]) 

        # Create the new empirical game by expanding it such that all policy combinations have an entry
        if self._empirical_game == None:
            self._empirical_game = {tuple([0] * self._num_players): {"fixed": np.nan, "no_halt": np.nan, "non_halt_proportion": np.nan}}
        else:
            new_empirical_game = {tuple(strat): {"fixed": np.nan, "no_halt": np.nan, "non_halt_proportion": np.nan} for strat in itertools.product(list(range(num_strategies)), repeat=self._num_players)}
            for k, v in self._empirical_game.items():
                new_empirical_game[k] = v 
            self._empirical_game = new_empirical_game

        print("Updating empirical game.")
        # For each of the possible policy combinations, see if there is an entry. If there isn't fill it in
        for chosen_strategies in itertools.product(list(range(num_strategies)), repeat=self._num_players):
            if all([type(v) != UtilityDistribution for v in self._empirical_game[tuple(chosen_strategies)].values()]):

                # i should be the player. j should be the strategy player i has chosen
                policies = [self._strategy_set[i][j] for i, j in enumerate(chosen_strategies)]

                # We use a context manager so that we can support an arbitrary number of contexts (policies) in the game
                all_graphs = [p.get_graph() for p in policies] + [self._model.get_frozen_graph()]
                context_managers = [tf.Session(graph=g) for g in all_graphs]

                # Simulation loop
                with ExitStack() as stack:                    
                    # Open all computational graphs for all policies and model
                    sessions = [stack.enter_context(cm) if cm != None else None for cm in context_managers]
                    model_session = sessions[-1]
                    policy_sessions = sessions[:-1]
                    
                    # The optimistic player is the player whose strategy was most recently added
                    optimistic_player = [i for i, j in enumerate(chosen_strategies) if j == (len(self._strategy_set[i]) - 1)][0]

                    # We want to retrieve the returns from ALL halt types for future evaluation. We determine which halt return distribution in update_profile
                    # all_halt_types = ["fixed", "optimistic"] if self._optimistic_distribution_initialized else ["fixed"]
                    for halt_type in ["no_halt", "fixed"]:
                        # Provide a rollout to estimations
                        print("Estimating utility for strategy {} with halt type {}.".format(chosen_strategies, halt_type))
                        all_returns, halted_trajectories, start = [], 0, time.time()
                        for _ in range(num_simulations):
                            halt_threshold_curr = 1e9 if halt_type == "no_halt" else halt_threshold
                            returns, _, halt, _, _ = self.rollout_model(policies, policy_sessions, model_session, halt_threshold_curr, True, kappa, optimistic_player=optimistic_player)
                            if halt:
                                halted_trajectories += 1
                            all_returns.append(returns)
                        print("Finished utility for {} in {} seconds with {} proportion of halted trajectories. \n".format(chosen_strategies, time.time() - start, halted_trajectories / num_simulations))
                        self._empirical_game[tuple(chosen_strategies)][halt_type] = UtilityDistribution(all_returns)
                        if halt_type == "fixed":
                            self._empirical_game[tuple(chosen_strategies)]["non_halt_proportion"] = 1 - (halted_trajectories / num_simulations)

                # # Given our simulations, we populate our empirical game with bootstrapped sample means
                # for halt_index, halt_type in enumerate(all_halt_types):
                #     # Represents the distribution of sample means possible of our utility 
                #     mean_distribution = [np.zeros(self._num_players) for _ in range(num_simulations)]

                #     # For each of the return samples 
                #     for r_list_all_halts, p_list_all_halts in zip(all_returns, all_probs):

                #         # Query for the halt type in question
                #         r_list, p_list = r_list_all_halts[halt_index], p_list_all_halts[halt_index]

                #         # Sample all points relevant to this rollout for computational ease
                #         choices = np.random.choice(r_list.shape[0], p=p_list, size=num_simulations)
                #         rets = np.take(r_list, choices, axis=0)

                #         # Add this rollout's "contribution" or to the sample mean to our distribution
                #         mean_distribution += rets * (1.0 / num_simulations)

                #     # Update the empirical game for that particular strategy and halt type
                #     self._empirical_game[tuple(chosen_strategies)][halt_type] = UtilityDistribution(mean_distribution)

        # If symmetric, only one player's strategy set needs to be stored
        if self._symmetric:
            self._strategy_set = self._strategy_set[0]
    
    def update_profile(self, meta_strategy_solver, halt_reward_type, parameters={}):
        """ Invoke the specified MSS with the corresponding parameters"""
        # Reconstruct the matrix representation of the game using our dictionary
        strategy_set = [self._strategy_set for _ in range(self._num_players)] if self._symmetric else self._strategy_set
        meta_games = []

        # Populate the metagame matrix with Nan values
        for _ in range(self._num_players):
            meta_game_shape = [len(player_strategy_set) for player_strategy_set in strategy_set]
            curr_player_meta_game = np.empty(meta_game_shape)
            curr_player_meta_game[:] = np.nan
            meta_games.append(curr_player_meta_game)
        
        # For all of the pure strategies, populate the meta game with utilities 
        for pure_strategy, utilities_distribution in self._empirical_game.items():

            # If halt_reward_type is fixed, we always use the fixed distribution
            reward_type = "fixed"
            # if halt_reward_type == "optimistic":
            #     # If s was trained optimistically, all other strategies must have index < s to use optimistic distribution
            #     # (this also implies it won't be matched with optimistic strategies trained on the same iteration)
            #     for s in self._iterations_with_optimistic_strategy:
            #         num_strategies_before_s = sum([1 if s_prime < s else 0 for s_prime in pure_strategy])
            #         if num_strategies_before_s == len(pure_strategy) - 1:
            #             reward_type = "no_halt"
            #             break

            expected_utilities = utilities_distribution[reward_type].expected_value

            for p in range(self._num_players):
                meta_games[p][pure_strategy] = expected_utilities[p]                

        # Run the chosen meta-strategy solver 
        if meta_strategy_solver.lower() == "rrd":
            result = projected_replicator_dynamics.regularized_replicator_dynamics(
                meta_games, regret_lambda=parameters["regret_lambda"], symmetric=self._symmetric, unused_kwargs={}) 
        elif meta_strategy_solver.lower() == "expectiregret_rrd":
            result, lowest_regret = None, np.inf
            for _ in range(parameters["bootstrap_rd_iterations"]):
                if self._symmetric:
                    initial_strategies = [np.random.choice(range(1, 11)) for _ in range(meta_games[0].shape[0])]
                    initial_strategies = [initial_strategies for _ in range(self._num_players)]
                else:
                    initial_strategies = [[np.random.choice(range(1,11)) for _ in range(meta_games[0].shape[k])] for k in range(self._num_players)]
                initial_strategies = [np.array(mixture) / np.sum(mixture) for mixture in initial_strategies]
                curr_result = projected_replicator_dynamics.regularized_replicator_dynamics(
                    meta_games, prd_initial_strategies=initial_strategies, regret_lambda=parameters["regret_lambda"], symmetric=self._symmetric, unused_kwargs={}) 
                regret = calculate_expected_regret(curr_result, self._empirical_game, parameters["expected_regret_estimation_iterations"], self._num_players)
                print("Found profile with expected regret of {}".format(regret))
                if regret < lowest_regret:
                    result = curr_result
                    lowest_regret = regret
            print("Settled on profile {} with expected regret of {}.".format(result, lowest_regret))
        else:
            raise NotImplementedError

        # Check if the all most recent best response appropriately created a best-response. If it didn't, assume we have to set self._optimistic to True. Otherwise, False.
        failed_to_best_respond = []
        if self._symmetric:
            profile = [self._profile for _ in range(self._num_players)]

        # Checking deviation payoffs
        for p in range(self._num_players):
            prev_profile_with_padding = [np.hstack([np.array(player_profile), np.zeros(1)]) for player_profile in profile]

            # meta_games is [players, shape_empirical_game] but we must swtich to [shape_empirical_game, players]
            swapped_meta_game = np.transpose(np.array(meta_games), list(range(1, len(np.array(meta_games).shape))) + [0])

            # It failed to best respond if the most recently added strategy's deviation payoff is not highest given the previous profile
            deviation_payoffs, _ = get_curr_and_deviation_payoffs(self._empirical_game.keys(), swapped_meta_game, prev_profile_with_padding, p)
            failed_to_best_respond.append(np.argmax(deviation_payoffs) != (np.shape(deviation_payoffs)[0] - 1))

        print("Failed to best respond? ", failed_to_best_respond)
        self._optimistic_iteration = all(failed_to_best_respond)
        self._profile = result 

        # Symmetric games only track one player's profile
        if self._symmetric:
            self._profile = self._profile[0]
        
        if self._verbose:
            if self._num_players <= 2:
                print("Displaying current meta-game: ")
                print('')
                self.display_meta_game(meta_games)
                print("\n")
            print("New profile: ", self._profile)
            print("\n")

    
    def display_meta_game(self, meta_games):
        # if self._symmetric:
        #     print("This game is symmetric. Displaying row player's payoff matrix. \n")
        #     meta_games_to_display = [meta_games[0]]
        # else:
        meta_games_to_display = meta_games

        for player, meta_game in enumerate(meta_games_to_display):
            print("\nPlayer {} Payoff Matrix: ".format(player))
            for row in meta_game:
                print(" ".join(["{0:6.3f}".format(val) for val in row]))

    def save_current_iteration_data(self, iteration_number):
        if self._symmetric:
            profile_to_save = [self._profile for _ in range(self._num_players)]
        meta_game_to_save = self._empirical_game

        # NOTE: Not saving the meta_game at the moment because we do not need it. Previously needed it for final_solution generation, but we are no longer doing that 
        # NOTE: Furthermore, saving the meta game takes a dispropotionate amount of memory for some reason. Look into that
        # with open(self._save_path + 'meta_game_iteration_{}.pkl'.format(iteration_number), 'wb') as f:
        #     pickle.dump(meta_game_to_save, f)
        
        with open(self._save_path + 'profile_iteration_{}.pkl'.format(iteration_number), 'wb') as f:
            pickle.dump(profile_to_save, f)
    
    def get_future_reward_distribution(self, t, halt_reward_type, player):
        """
        Given a time t in which this trajectory halted, calculate the distribution of returns possible for all players given a trajectory was halted
        """
        # if halt_reward_type == "pessimistic":
        #     return self._pessimistic_return_distribution[t-1], self._pessimistic_return_probabilities[t-1]
        if halt_reward_type == "optimistic":
            return self._optimistic_distribution[player][t-1], self._optimistic_probabilities[player][t-1]
        else:
            raise NotImplementedError

    def rollout_model(self, policies, policy_sessions, model_session, halt_threshold, is_evaluation=False, kappa=5, optimistic_player=0, apply_incentive=False, incentive_session=None):
        """
        Execute one rollout using the trained dynamics model. Should be the only entry point to generating dynamics model guided rollouts. 

        policies: len(self._num_players) list of policy objects (requires step method) to execute one rollout with
        policy_sessions: len(self._num_players) list of sessions that contain computational graphs for each policy
        model_session: a single session containing computational graph for the model
        is_evaluation: toggles whether we are training one of the policies 
        kappa: MoREL penalty if we are calculating the for a "fixed" halt reward type 
        optimistic_player: if we are training or evaluating an optimistic player, which player is it?
        """
        ################################ Initializations ######################################
        # Sample from the trajectories to find a starting point
        sampled_trajectory = np.random.choice(self._data)
        sampled_start = sampled_trajectory[0]
       
        # Gather information for dynamics model
        global_state = sampled_start.global_state
        relevant_players = sampled_start.relevant_players

        # Initialize Step for action generation and observations for information state generation (using the game-specific module) 
        if self._is_turn_based:
            # For turn based games, observations from ALL players carry the information needed to reconstruct info_states for each player
            starting_player = relevant_players[0]
            all_observations = [[sampled_start.info_states[relevant_players[0]]] for p in range(self._num_players)] 
            info_state = self._true_state_extractor.observations_to_info_state(all_observations[relevant_players[0]])
            
            # Create the Step object to pass into policies
            curr_step = Step(info_state=info_state, reward=None, is_terminal=False, halted=False, acting_players=relevant_players, 
                            global_state=global_state, legal_actions_mask=sampled_start.legal_actions_masks[relevant_players[0]])
        else:
            # all_observations = [[curr_step.info_state[p]] for p in env.num_players]
            raise NotImplementedError

        # Trackers for terminals, number of steps, and returns
        steps, episode_length, done, rollout, actions = [0] * self._num_players, 0, False, [curr_step], []
        returns = [0.0 for _ in range(self._num_players)]

        ############################## Initializations End ###################################
    
        ############################### Rollout Loop #########################################
        while not done:
            if self._is_turn_based:
                # Get the current player, determine their action, and increment the number of steps they've taken this episode
                curr_player = curr_step.acting_players[0]
                action = [policies[curr_player].step(curr_step, curr_player, policy_sessions[curr_player], is_evaluation=is_evaluation)[0]]
                steps[curr_player] += 1
                
                # Pass action and state into the model. Output next state, reward, all player's new observations and new legal actions, and whether state is terminal
                next_state, reward, player_next_observations, next_legal_action_masks, done, halt = self._model.get_next_step(curr_step.global_state, action, halt_threshold, frozen_session=model_session)

                # Increment episode length
                episode_length += 1

                # NOTE: Assumption is players always alternate. May not always be true!
                next_player = (relevant_players[0] + 1) % self._num_players 

                # Append new observations 
                for p in range(self._num_players):
                    all_observations[p].append(player_next_observations[next_player])

                # Construct the next step for the upcoming player
                info_state = self._true_state_extractor.observations_to_info_state(all_observations[next_player])

                # We do this in the loop because we need our Step object to include "done" information
                done = done or episode_length >= self._max_episode_length

                # If we halted, then apply our HALT-utility distribution
                if halt:
                    reward = np.reshape(np.array([-kappa for _ in range(self._num_players)]), (1, -1))
                    returns = [r - kappa for r in returns] 
                # Otherwise, simply apply whichever reward was returned by our model
                else:
                    # if apply_incentive:
                    #     added_incentive = self._incentive_model.predict(global_state, action, curr_player, frozen_session=incentive_session)
                    #     reward[0][curr_player] += self._incentive_alpha * added_incentive
                    returns = [r + reward[0][p] for p, r in enumerate(returns)]
                
                # Reassign curr_step. If we are "done," then only reward is used.
                curr_step = Step(info_state=info_state, reward=reward, is_terminal=done, halted=halt, legal_actions_mask=next_legal_action_masks, acting_players=[next_player], global_state=next_state)
                rollout.append(curr_step)
                actions.append([action])  # make a list for generalization to simultaneous games
            else:
                steps = [steps[0] + 1 for _ in range(self._num_players)]
                episode_length += 1
                raise NotImplementedError
            
        # Last time step to account for terminal state
        for p in range(self._num_players):
            policies[p].step(curr_step, p, policy_sessions[p], is_evaluation=is_evaluation)  
        ############################# Rollout Loop End ####################################

        return returns, np.array(steps), halt, rollout, actions
