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
from open_spiel.python.rl_environment import StepType
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
        self._data = np.array(data) # [num_players, num_ensemble]
    
    @property
    def expected_value(self):
        return np.mean(self._data, axis=1) # np.mean(self._data, axis=0)
    
    @property
    def lower_bound(self):
        return np.min(self._data, axis=1)
    
    @property 
    def upper_bound(self):
        return np.max(self._data, axis=1)

    def create_permuted_copy(self, permutation):
        new_data = np.array([self._data[p] for p in permutation])
        return UtilityDistribution(new_data)
########## Utility Distribution Class for Ease of Representation End ###########

########## Named Tuple Representing a Noisy Start ###########
Start = collections.namedtuple(
    "Start",
    "info_states legal_actions_masks relevant_players global_state")
######## Named Tuple Representing a Noisy Start End #########

class OfflineModelBasedPSRO:
    def __init__(self, dataset, symmetric, is_turn_based, num_players, num_actions, initial_strategies, model_args, 
                 tf_model_management_module, save_path, true_state_extractor, max_episode_length, verbose=False):
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
            env_variance: the max amount of gaussian noise variance to apply to start states
            verbose: toggles display
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
        self._indicator_rounding = model_args["indicator_rounding"]
        
        self._verbose = verbose
        self._empirical_game = None 
        self._save_path = save_path if save_path[-1] == "/" else save_path + "/"
        self._num_rollouts_for_noisy_starts = 100
        
        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)
        ################################ Initializations End ####################################
        
        ################################ Model Training #########################################
        with open(self._save_path+WorldModelDeterministic.model_arg_name, 'wb') as f:
            pickle.dump(model_args, f)
        
        with tf.Session() as sess:
            model_args["session"] = sess
            # Model Type Deterministic or Stochastic
            if model_args["model_type"] == "deterministic":
                self._model = WorldModelDeterministic(model_args["state_size"], model_args["action_size"], model_args) # Simple MLP
            else:
                raise NotImplementedError # World model MDN
            
            # Train and Save Model
            losses, reward_losses = self._model.train(num_gradient_steps=model_args["training_steps"])
            tf_model_management_module.save(sess, self._save_path + "model.ckpt")
        
        # Plot generation and saving for dynamics
        plt.plot(range(len(losses)), losses)
        plt.savefig(self._save_path+'model_loss.jpg')
        plt.clf()

        # Plot generation and saving for reward
        plt.plot(range(len(reward_losses)), reward_losses)
        plt.savefig(self._save_path+'reward_model_loss.jpg')
        plt.clf()
        
        # Outside of the session, load and freeze the model.
        self._model.freeze(tf_model_management_module, self._save_path)
        ############################## Model Training End #######################################

        #################### Strategy and Profile Initializations ####################
        if self._symmetric:
            if len(initial_strategies) > 1 or len(initial_strategies) == 0:
                logging.error("{} initial strategies were passed in for symmetric game.".format(len(initial_strategies)))
                raise Exception 
            if len(initial_strategies) == 1:
                self._strategy_set = initial_strategies 
                self._profile = [1.0]
        else:
            if len(initial_strategies) != self._num_players:
                logging.error("Number of strategies in initial_strategies {} does not match the number of players {}".format(len(initial_strategies), self._num_players))
                raise Exception 
            if len(initial_strategies) == self._num_players:
                self._strategy_set = [[initial_strategies[i]] for i in range(self._num_players)]
                self._profile = [[1.0] for i in range(self._num_players)]
        self._env_profile = [1.0]
        ################## Strategy and Profile Initializations End ##################

    # TODO: Modify this so that it takes in a parameter alpha, determining the probability that we will explicitly optimize for deviation coverage
    def train_and_add_strategy(self, response_parameters, training_player, num_training_steps, alpha, mopo_penalty):
        """
            response_parameters: describes the type of RL algorithm and parameters for it that will be used to train new algorithm
            training_player: player index whose strategy set we are adding to
            num_training_steps: number of environment interactions the training_player will receive during training
        """
        ############################## Initializations ##############################
        # If symmetric, then strategy set and profile are associated with only one of the players. Repeat for each player for coding simplicity
        if self._symmetric:
            self._strategy_set = [self._strategy_set for _ in range(self._num_players)]
            self._profile = [self._profile for _ in range(self._num_players)]

        # Get computational graphs for each of the frozen policies/models and load them into our context manager
        all_graphs = [[pol.get_graph() for pol in player_set] for player_set in self._strategy_set]
        context_managers = [[tf.Session(graph=g) for g in player_set] for player_set in all_graphs]
        ############################ Initializations End ############################

        ############################ Training Loop ##################################
        # Load all the sessions and graphs from the previously frozen policies
        with ExitStack() as stack:
            # Load policy graphs 
            sessions = [[stack.enter_context(cm) if cm != None else None for cm in cm_list] for cm_list in context_managers]
            
            # Load model graph
            model_session = stack.enter_context(tf.Session(graph=self._model.get_frozen_graph()))

            # Create a new computational graph for our new policy
            new_rl_training_session = stack.enter_context(tf.Session(graph=tf.Graph()))
            sessions[training_player].append(new_rl_training_session)


            # Initialize new policy and append
            new_policy = DQN(new_rl_training_session, graph=new_rl_training_session.graph, reward_normalizer=self._model._reward_normalizer, **response_parameters)
            new_policy.set_id(len(self._strategy_set[training_player]))

            self._strategy_set[training_player].append(new_policy)

            # Training loop for new policy
            curr_training_steps = 0
            aggregate_returns, aggregate_discrepancy = [], []
            while curr_training_steps < num_training_steps:
                # Sample from profiles to get response target
                policies, policy_sessions = [], []

                # Otherwise, keep this and set real reward weight to 1
                for p in range(self._num_players):
                    if p == training_player:
                        policies.append(new_policy)
                        policy_sessions.append(new_rl_training_session)
                    else:
                        if np.random.random() < alpha:
                            strategy_choice = np.random.choice(list(range(len(self._profile[p]))), p=[1.0 / len(self._profile[p])] * len(self._profile[p]))
                            real_reward_weight = 0
                            policies.append(UniformRandomPolicy(state_size=1, num_actions=self._num_actions))
                            policy_sessions.append(None)
                        else:
                            strategy_choice = np.random.choice(list(range(len(self._profile[p]))), p=self._profile[p])
                            real_reward_weight = 1
                            policies.append(self._strategy_set[p][strategy_choice])
                            policy_sessions.append(sessions[p][strategy_choice])

                # Do one rollout using the trained model. Set is_evaluation to False to train new_policy
                returns, steps, _, _, max_discrepancy, _ = self.rollout_model(policies, policy_sessions, model_session, is_evaluation=False, mopo_penalty=mopo_penalty, real_reward_weight=real_reward_weight)

                aggregate_returns.append(returns[training_player])
                aggregate_discrepancy.append(max_discrepancy)
                
                # Track the number of steps for the currently training player
                curr_training_steps += steps[training_player] 

            window = 1000
            averaged_over = [sum(aggregate_returns[i:i+window]) / window for i in range(len(aggregate_returns) - window + 1)]
            plt.plot(list(range(len(averaged_over))), averaged_over)

            # To distinguish between different players, we create player-specific directories for their policies.
            # In symmetric games, there will only be a folder for player 0 (because strategies are shared)
            policy_save_path = self._save_path + "player_{}/".format(training_player)
            if not os.path.exists(policy_save_path):
                os.makedirs(policy_save_path)
                
            plt.savefig(policy_save_path+'policy_model_rewards_policy_{}_player_{}.jpg'.format(len(self._strategy_set[training_player]), training_player))
            plt.clf()

            averaged_over = [sum(aggregate_discrepancy[i:i+window]) / window for i in range(len(aggregate_discrepancy) - window + 1)]
            plt.plot(list(range(len(averaged_over))), averaged_over)
                
            plt.savefig(policy_save_path+'policy_model_discrepancy_policy_{}_player_{}.jpg'.format(len(self._strategy_set[training_player]), training_player))
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

    def update_empirical_game(self, num_simulations):
        """
        Update entire empirical game with utility distributions for each entry.
        
        num_simulations: number of model rollouts to use to best estimate utilities
        """
        ############# Symmetry Handling #############
        # If symmetric, then strategy sets are for one player. For coding simplicity, make copies 
        if self._symmetric:
            self._strategy_set = [self._strategy_set for _ in range(self._num_players)]
        num_strategies = len(self._strategy_set[0]) 
        ########### Symmetry Handling End ###########
        

        # Create the new empirical game by expanding it such that all policy combinations have an entry
        ############# Empirical Game Dictionary Creation #############
        if self._empirical_game == None:
            self._empirical_game = {tuple([0] * (self._num_players)): {"utility": np.nan, "discrepancy": np.nan}}
        else:
            new_empirical_game = {tuple(strat): {"utility": np.nan, "discrepancy": np.nan} for strat in itertools.product(*([list(range(num_strategies))] * self._num_players))}
            for k, v in self._empirical_game.items():
                new_empirical_game[k] = v 
            self._empirical_game = new_empirical_game
        ########### Empirical Game Dictionary Creation End ###########

        print("Updating empirical game.")
        ############# Filling in Empirical Game Entries #############
        for chosen_strategies in itertools.product(*([list(range(num_strategies))] * self._num_players)):
            if all([type(v) != UtilityDistribution for v in self._empirical_game[tuple(chosen_strategies)].values()]):
                # i should be the player. j should be the strategy player i has chosen
                policies = [self._strategy_set[i][j] for i, j in enumerate(chosen_strategies)]

                # We use a context manager so that we can support an arbitrary number of contexts (policies) in the game
                all_graphs = [p.get_graph() for p in policies] + [self._model.get_frozen_graph()]
                context_managers = [tf.Session(graph=g) for g in all_graphs]

                # Simulation loop
                with ExitStack() as stack:                    
                    ############# Session Initializations #############
                    sessions = [stack.enter_context(cm) if cm != None else None for cm in context_managers]
                    model_session = sessions[-1]
                    policy_sessions = sessions[:-1]
                    ########### Session Initializations End ###########
                    
                    ############# Estimating Utilities for Players Using True Policies #############
                    print("Estimating utility for strategy {}.".format(chosen_strategies))
                    all_returns, start, total_discrepancy, all_reward_ensembles = np.zeros(self._num_players), time.time(), 0, np.zeros((self._num_players, self._model._ensemble_size))
                    for _ in range(num_simulations):
                        # We set the mopo penalty for empirical game updates to 0.
                        returns, _, _, _, discrepancy, reward_ensemble = self.rollout_model(policies, policy_sessions, model_session, is_evaluation=True, mopo_penalty=0)
                        all_returns += np.array(returns)
                        total_discrepancy += discrepancy
                        all_reward_ensembles += reward_ensemble
                    averaged_returns = all_returns / num_simulations 
                    averaged_discrepancy = total_discrepancy / num_simulations
                    averaged_reward_ensemble = all_reward_ensembles / num_simulations
                    print("Finished utility for {} in {} seconds with raw averaged returns of {}, total discrepancy of {}, lower bound {} and upper bound {}.".format(chosen_strategies, time.time() - start, averaged_returns, averaged_discrepancy, 
                                                                                                                                                                        np.min(averaged_reward_ensemble, axis=1), np.max(averaged_reward_ensemble, axis=1)))
                    ########### Estimating Utilities for Players Using True Policies End ###########
                    
                    ############# Populating Metagame with Utilities Estimates #############
                    if self._symmetric:
                        # Average over payoffs for any players that played the same strategy 
                        unique_strategies = set(chosen_strategies)
                        strategy_to_payoff = {}
                        for s in unique_strategies:
                            averaged_payoff = np.mean([averaged_reward_ensemble[i] for i in range(self._num_players) if chosen_strategies[i] == s], axis=0)
                            strategy_to_payoff[s] = averaged_payoff
                        distribution = UtilityDistribution(np.array([strategy_to_payoff[s] for s in chosen_strategies]))
                        for permutation in itertools.permutations(list(range(len(chosen_strategies))), len(chosen_strategies)):
                            self._empirical_game[tuple([chosen_strategies[new_player_assignment] for new_player_assignment in permutation])]["utility"] = distribution.create_permuted_copy(list(permutation))
                            self._empirical_game[tuple([chosen_strategies[new_player_assignment] for new_player_assignment in permutation])]["discrepancy"] = averaged_discrepancy
                        # Permute over the payoffs to update the empirical game
                    else:
                        self._empirical_game[tuple(chosen_strategies)]["utility"] = UtilityDistribution(averaged_returns)
                        self._empirical_game[tuple(chosen_strategies)]["discrepancy"] = averaged_discrepancy
                    ########### Populating Metagame with Utility Estimates End ###########
        ########### Filling in Empirical Game Entries End ###########

        # If symmetric, only one player's strategy set needs to be stored
        if self._symmetric:
            self._strategy_set = self._strategy_set[0]
    
    def update_profile(self, meta_strategy_solver, parameters={}):
        """ Invoke the specified MSS with the corresponding parameters. """

        ###################### Matrix Representation Reconstruction ###################################
        strategy_set = [self._strategy_set for _ in range(self._num_players)] if self._symmetric else self._strategy_set
        meta_games = []
        lower_bound_meta_games = []
        upper_bound_meta_games = []

        # Populate the metagame matrix with Nan values
        for _ in range(self._num_players):
            meta_game_shape = [len(player_strategy_set) for player_strategy_set in strategy_set]
            curr_player_meta_game = np.empty(meta_game_shape)
            curr_player_meta_game[:] = np.nan
            meta_games.append(curr_player_meta_game)
            lower_bound_meta_games.append(np.copy(curr_player_meta_game))
            upper_bound_meta_games.append(np.copy(curr_player_meta_game))
        #################### Matrix Representation Reconstruction End #################################
        
        #################### Populating Meta Game with Utilities #################################
        for pure_strategy, utilities_distribution in self._empirical_game.items():
            expected_utilities = utilities_distribution["utility"].expected_value
            for p in range(self._num_players):
                meta_games[p][pure_strategy] = expected_utilities[p]     
            
            lower_bound = utilities_distribution["utility"].lower_bound
            for p in range(self._num_players):
                lower_bound_meta_games[p][pure_strategy] = lower_bound[p]

            upper_bound = utilities_distribution["utility"].upper_bound
            for p in range(self._num_players):
                upper_bound_meta_games[p][pure_strategy] = upper_bound[p]
        ################## Populating Meta Game with Utilities End ###############################           
 
        #################### Invoke Meta-Strategy Solver #################################
        if meta_strategy_solver.lower() == "rrd":
            result = projected_replicator_dynamics.regularized_replicator_dynamics(
                meta_games, regret_lambda=parameters["regret_lambda"], symmetric=self._symmetric, symmetric_players=[[i for i in range(self._num_players)]], unused_kwargs={}) 
            
            nash_result = projected_replicator_dynamics.regularized_replicator_dynamics(
                meta_games, regret_lambda=1e-5, symmetric=self._symmetric, symmetric_players=[[i for i in range(self._num_players)]], unused_kwargs={}) 
            
        elif meta_strategy_solver.lower() == "r3d":
            result, training_curve = projected_replicator_dynamics.robust_regularized_replicator_dynamics(
                lower_bound_meta_games, upper_bound_meta_games, regret_lambda=parameters["regret_lambda"],
                symmetric=self._symmetric, symmetric_players=[[i for i in range(self._num_players)]], entropy_calculation_players=[i for i in range(self._num_players)],
                defaults=[None for _ in range(self._num_players)]) 
            
            plt.clf()
            plt.plot(list(range(len(training_curve))), training_curve)
            plt.savefig(self._save_path + 'r3d_curve_{}.png'.format(len(self._strategy_set)))
            plt.clf()

            nash_result, _ = projected_replicator_dynamics.robust_regularized_replicator_dynamics(
                lower_bound_meta_games, upper_bound_meta_games, regret_lambda=0, 
                symmetric=self._symmetric, symmetric_players=[[i for i in range(self._num_players)]], 
                defaults=[None for _ in range(self._num_players)])
            
        else:
            raise NotImplementedError
        self._profile = [result[i] for i in range(self._num_players)]
        self._final_profile = [nash_result[i] for i in range(self._num_players)]
        ################## Invoke Meta-Strategy Solver End ###############################

        ################## Symmetry and Display Handling ##################
        if self._symmetric:
            self._profile = self._profile[0]
            self._final_profile = self._final_profile[0]
        
        if self._verbose:
            # print("Meta-game displays are not enabled.\n")
            print("Displaying current meta-game: \n")
            self.display_meta_game(meta_games[:-1])

        print("\nNew profile: {} ".format(self._profile))
        print("\nNash solution: {} \n\n".format(self._final_profile))
        ################ Symmetry and Display Handling End ################

    
    def display_meta_game(self, meta_games):
        ##################  Display Metagame by Row ##################
        meta_games_to_display = meta_games
        for player, meta_game in enumerate(meta_games_to_display):
            print("\nPlayer {} Payoff Matrix: ".format(player))
            for row in meta_game:
                print(" ".join(["{0:6.3f}".format(val) for val in row]))
        #################  Display Metagame by Row ENd ################

    def save_current_iteration_data(self, iteration_number):
        ################# Saving Profile and Meta-Game ################
        if self._symmetric:
            profile_to_save = [self._profile for _ in range(self._num_players)]
            final_profile_to_save = [self._final_profile for _ in range(self._num_players)]

        # with open(self._save_path + 'meta_game_iteration_{}.pkl'.format(iteration_number), 'wb') as f:
        #     pickle.dump(self._empirical_game, f)
        
        with open(self._save_path + 'profile_iteration_{}.pkl'.format(iteration_number), 'wb') as f:
            pickle.dump(profile_to_save, f)

        with open(self._save_path + 'final_solution_{}.pkl'.format(iteration_number), 'wb') as f:
            pickle.dump(final_profile_to_save, f)
        ############### Saving Profile and Meta-Game End ##############
    
    def save_empirical_game(self):
        with open(self._save_path + 'model_meta_game.pkl', 'wb') as f:
            pickle.dump(self._empirical_game, f)

    def rollout_model(self, policies, policy_sessions, model_session, manual_start=None, is_evaluation=False, mopo_penalty=1, real_reward_weight=1):
        """
        Execute one rollout using the trained dynamics model. Should be the only entry point to generating dynamics model guided rollouts. 

        policies: len(self._num_players) list of policy objects (requires step method) to execute one rollout with
        policy_sessions: len(self._num_players) list of sessions that contain computational graphs for each policy
        model_session: a single session containing computational graph for the model
        is_evaluation: toggles whether we are training one of the policies 
        """
        ################################ Initializations ######################################
        # Sample from the trajectories to find a starting point
        if manual_start:
            sampled_start = manual_start 
        else:
            sampled_start = np.random.choice(self._data)[0] 
       
        # Gather information for dynamics model
        global_state = sampled_start.global_state
        relevant_players = sampled_start.relevant_players

        # Initialize Step for action generation and observations for information state generation (using the game-specific module) 
        if self._is_turn_based:
            # For turn based games, observations from ALL players carry the information needed to reconstruct info_states for each player
            starting_player = relevant_players[0]
            all_observations = [[sampled_start.info_states[starting_player]] for p in range(self._num_players)] 
            info_state = self._true_state_extractor.observations_to_info_state(all_observations[starting_player])
            
            # Create the Step object to pass into policies
            curr_step = Step(info_state=info_state, reward=None, is_terminal=False, acting_players=relevant_players, 
                            global_state=global_state, legal_actions_mask=sampled_start.legal_actions_masks[starting_player])
        else:
            # all_observations = [[curr_step.info_state[p]] for p in env.num_players]
            raise NotImplementedError

        # Trackers for terminals, number of steps, and returns
        steps, episode_length, done, rollout, actions = [0] * self._num_players, 0, False, [curr_step], []
        returns = [0.0 for _ in range(self._num_players)]
        reward_ensemble_total = np.zeros((self._num_players, self._model._ensemble_size))
        discrepancy_total = 0

        ############################## Initializations End ###################################
    
        ############################### Rollout Loop #########################################
        while not done:
            if self._is_turn_based:
                # Get the current player, determine their action, and increment the number of steps they've taken this episode
                curr_player = curr_step.acting_players[0]

                if type(is_evaluation) == list:
                    curr_eval = is_evaluation[curr_player]
                else:
                    curr_eval = is_evaluation 

                action = [policies[curr_player].step(curr_step, curr_player, policy_sessions[curr_player], is_evaluation=curr_eval)[0]]

                steps[curr_player] += 1
                
                # Pass action and state into the model. Output next state, reward, all player's new observations and new legal actions, and whether state is terminal
                next_state, reward, player_next_observations, next_legal_action_masks, done, max_prediction_discrepancy, reward_ensemble = self._model.get_next_step(curr_step.global_state, action, frozen_session=model_session)

                # Increment episode length
                episode_length += 1

                # NOTE: Assumption is players always alternate. May not always be true!
                next_player = (curr_player + 1) % self._num_players 

                # Append new observations 
                for p in range(self._num_players):
                    all_observations[p].append(player_next_observations[next_player])

                # Construct the next step for the upcoming player
                info_state = self._true_state_extractor.observations_to_info_state(all_observations[next_player])

                # We do this in the loop because we need our Step object to include "done" information
                done = done or (episode_length > self._max_episode_length)

                reward = np.reshape(np.array([real_reward_weight * reward[0][i] - mopo_penalty * max_prediction_discrepancy for i in range(self._num_players)]), (1, -1))
                reward_ensemble_total += reward_ensemble * real_reward_weight - mopo_penalty * max_prediction_discrepancy
                discrepancy_total += max_prediction_discrepancy
                returns = [r + real_reward_weight * reward[0][p] - mopo_penalty * max_prediction_discrepancy for p, r in enumerate(returns)]
                
                # Reassign curr_step. If we are "done," then only reward is used.
                curr_step = Step(info_state=info_state, reward=reward, is_terminal=done, legal_actions_mask=next_legal_action_masks, acting_players=[next_player], global_state=next_state)
                # rollout.append(curr_step)
                # actions.append([action])  # make a list for generalization to simultaneous games
            else:
                steps = [steps[0] + 1 for _ in range(self._num_players)]
                episode_length += 1
                raise NotImplementedError
            
        # Last time step to account for terminal state
        for p in range(self._num_players):
            if type(is_evaluation) == list:
                    curr_eval = is_evaluation[p]
            else:
                curr_eval = is_evaluation 
            policies[p].step(curr_step, p, policy_sessions[p], is_evaluation=curr_eval)  
        ############################# Rollout Loop End ####################################

        return returns, np.array(steps), [], [], discrepancy_total, reward_ensemble_total
