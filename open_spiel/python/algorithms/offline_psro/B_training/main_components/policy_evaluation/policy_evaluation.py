"""
This is dedicated to creating and updating a normal form game model using offline policy evaluations, covariate updates, and variance analysis.
"""

import collections
import time
from collections import defaultdict

import os
import matplotlib.pyplot as plt
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
from open_spiel.python.algorithms.offline_psro.utils.utils import generate_single_rollout, convert_trajectory_to_string, get_new_default_policy_info_dict, get_new_default_policy_evaluation_information



class OfflineNormalFormGame:
    def __init__(self, data, num_players, is_symmetric, apply_per_decision=False, apply_weighted=True):
        """
        data: Data should be a list consisting of trajectories. Each trajectory is a list of transitions. 

        Each Transition is a named tuple with fields (except for done and current players) that can be indexed using the player_id:
          [observations, actions, rewards, next observations, next actions, done, legal action masks, relevant players]

        Simultaneous games: each transition will have relevant players = [all player indices]
        Turn-based games: each transition will have relevant players = [curr player] 
          except for the last transition, where relevant players = [all players indices]
          since we need to account for terminal nodes for each of player
        """
        self._data = data
        self._data_setified = dict() # This is the "setified" dataset, where we map a string-representation of a trajectory to the corresponding trajectory
        self._is_symmetric = is_symmetric 
        self._num_players = num_players

        self._apply_per_decision = apply_per_decision 
        self._apply_weighted = apply_weighted

        self._strategies = [[] for _ in range(1 if self._is_symmetric else self._num_players)]
        self._empirical_game = [[] for _ in range(1 if self._is_symmetric else self._num_players)]

        # Map a single policy (can be either behavior or evaluation) to a dictionary
        # This dictioanry will map player to another dictionary
        # This dictionary will map trajectory index to a length 2 list: probabilities and corresponding timesteps
        self._cache_policy_probabilities = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) 

        # Maps a tuple of policies () -> trajectory index -> list of ratios 
        self._cache_is_weights = defaultdict(lambda: defaultdict(list))


    ##############################
    ### Empirical Game Updates  ## 
    ##############################
    def update_empirical_game(self, strategies):
        """
        A method that takes in new strategies to add and updates the normal form game to reflect the policy evaluations for the new entries.
        """
        return

    def add_strategies(self, strategies):
        """
        Adds strategies to the strategy set self._strategies. len(strategies) must be equal to len(self._strategies)
        where each element in strategies corresponds to strategies to be added to the respective player's strategy set 
        """
        # If is_symmetric, then len(strategies) must be 1
        if self._is_symmetric and len(strategies) > 1:
            logging.error(" Addition of strategies had improper lengths: symmetric games must only add strategies for 1 player. ")
        if len(strategies) != len(self._strategies):
            logging.error(" Addition of strategies had improper lengths: number of strategy sets added must match the number of players in game.")
        
        for strategy_set, strategies_to_add in zip(self._strategies, strategies):
            strategy_set.extend(strategies_to_add)

    ##############################
    # Empirical Game Updates End #
    ##############################

    #########################
    ## Importance Sampling ##
    #########################

    def compute_behavior_policy(self, data_indices, policy_class, observation_size, action_size):
        """
        Return the estimated behavior policy from a subset of the dataset where all policies are modeled by the same policy class

        data_indices: a list of indices specifying which indices of the dataset on which to estimate/train the behavior policy
        policy_class: a class generator that returns a policy instance a particular type (a policy instance that inherits from PolicyWrapper)
        observation_size: a list of integers indicating how many elements are in each player's observation space  
        action_size: a list of integers indicating how many elements are in each player's action space
        """

        data_subset = [self._data[i] for i in data_indices]
        behavior_policies = [policy_class(observation_size[p], action_size[p]) for p in range(len(self._num_players))]
        # NOTE: This assumes that the game is non-symmetric (otherwise we would pass in a list for players)
        for i, pol in enumerate(behavior_policies):
            pol.train(data=data_subset, players=[i])
        return behavior_policies

    def calculate_importance_sampled_evaluation(self, behavior_policies, evaluation_policies, indices=None):
        """
        Calculate the importance sampled policy evaluation estimates for evaluation_policies under the dataset approximately generated by behavior_policies

        behavior_policies: a list of policy instances (one-per-player) that characterize the dataset
        evaluation_policies: a list of policy instances (one-per-player) in which we want to evaluate returns or utilities for
        indices: a list of indices on which to evaluate the evaluation_policies. If None, then it will use the entire dataset. 
        """

        if type(indices) != np.ndarray:
            if type(indices) == list:
                indices = np.array(indices)
            else:
                logging.warning(" No indices passed into importance sampled evaluation. Defaulting to all provided datapoints. ") 
                indices = np.array(list(range(len(self._data))))

        curr_weights = self.importance_sampling_weights(behavior_policies, evaluation_policies, indices)

        # Apply weighted likelihood ratios if we so choose
        if self._apply_weighted:
            curr_weights = self.apply_weighted_likelihood_ratios(curr_weights)
        else:
            num_trajectories = indices.shape[0]
            curr_weights = [[weight / num_trajectories for weight in lst] for lst in curr_weights]

        # curr_weights is a list of lists: each list corresponds to a trajectory in the dataset. Each element in the list 
        # is the weight assigned to a reward within each trajectory.

        policy_evaluation = np.zeros(self._num_players)
        all_evaluations = []
        for i, trajectory_index in enumerate(indices):
            trajectory = self._data[trajectory_index]
            for j, transition in enumerate(trajectory):
                # NOTE: We cannot insert cache for individual trajectory return estimates because weighted_IS makes these estimates dependent on the sample
                curr = curr_weights[i][j] * np.array(transition.rewards)
                policy_evaluation += curr
                all_evaluations.append(curr)

        return policy_evaluation, all_evaluations
        

    def importance_sampling_weights(self, behavior_policies, evaluation_policies, indices=None):
        """
        Calculate the importance sampling weights for policy evaluation of the trajectories specified by indices under the behavior and evaluation policies.

        behavior_policies: a list of policy instances (one-per-player) that characterize the dataset
        evaluation_policies: a list of policy instances (one-per-player) in which we want to evaluate returns or utilities for
        indices: a list of indices on which to evaluate the evaluation_policies. If None, then it will use the entire dataset. 
        """

        # Given a N trajectories, compute likelihood ratios, applying per-decision/weighted options
        # Ratios is a list of length num_trajectories and each list contains the likelihood ratio of each timestep within the trajectory
        ratios = []
        if type(indices) != np.ndarray:
            if type(indices) == list:
                indices = np.array(indices)
            else:
                logging.warning(" No indices passed into importance sampled evaluation. Defaulting to all provided datapoints. ") 
                indices = np.array(list(range(len(self._data))))
    
        cache_policy_query = '_'.join([str(b.id) for b in behavior_policies] + [str(e.id) for e in evaluation_policies])
        for i in indices:
            t = self._data[i]

            # TODO: Check if index i already has an associated list of ratios in the cache. If it does, return it.
            if len(self._cache_is_weights[cache_policy_query][i]) > 0:
                ratios.append(self._cache_is_weights[cache_policy_query][i])
                continue 

            # Get the individual likelihood ratios (rho) for each timestep
            rho = self.compute_rho(i, behavior_policies, evaluation_policies)

            # Sanity check
            if len(t) != len(rho):
                logging.error(" Trajectory lengths do not match in likelihood ratio calculation. Recheck implementation. ")
                raise Exception(" Rip ")

            # Aggregate the relevant timesteps for each of the timesteps 
            curr_ratios = [rho for _ in range(len(rho))]


            # Apply per decision if we so choose 
            if self._apply_per_decision:
                curr_ratios = self.apply_per_decision(curr_ratios)


            # Calculate the importance sampling weights 
            curr_ratios = [np.prod(lst_of_ratios) for lst_of_ratios in curr_ratios]

            # Append to the list of ratios
            ratios.append(curr_ratios)
            self._cache_is_weights[cache_policy_query][i] = curr_ratios

        return ratios 

    def compute_rho(self, trajectory_index, behavior_policies, evaluation_policies):
        """
        Compute rho, which is a list of length(trajectory) where each element is the SINGLE likelihood ratio (not product) 
            associated with each particular timestep, for a specified trajectory.
        This implementation batches queries using policies.probabilities_with_actions to minimize the number of passes to the policy.

        We assume that the behavior policy is fully modeled in decentralized space

        trajectory_index: which trajectory in the dataset to compute rho value sfor
        behavior_policies: a list of policy instances (one-per-player) that characterize the dataset
        evaluation_policies: a list of policy instances (one-per-player) in which we want to evaluate returns or utilities for

        """
        # Compute the rho values with minimal passes to the policy. Batch queries using policies.probabilities_with_actions
        # Note that rho corresponds to the individual likelihood ratios each timestep (not multiplied across timesteps)
        # We assume that the behavior policy is fully modeled in decentralized space (assume joint space provides no more info)

        # Initialize list of rho values to 1 
        trajectory = self._data[trajectory_index]
        rho = [1 for _ in range(len(trajectory))]

        # Note that the corresponding transitions per player should be the same regardless of whether we are querying for evaluation or behavior policies
        evaluation_probabilities, behavior_probabilities, corresponding_transitions = [], [], []
        for query_player, eval_policy in enumerate(evaluation_policies):
            info = self.get_policy_probabilities(trajectory_index, eval_policy, query_player)  # probs, corresponding transitions 
            evaluation_probabilities.append(info[0])
            corresponding_transitions.append(info[1])
        for query_player, behavior_policy in enumerate(behavior_policies):
            info = self.get_policy_probabilities(trajectory_index, behavior_policy, query_player)
            behavior_probabilities.append(info[0])

        # Query for each of the players
        for player in range(self._num_players):
            evaluation_probs = evaluation_probabilities[player]
            behavior_probs = behavior_probabilities[player]

            # Evaluation and behavior probs are of shape N x 1 matrix where N is the number of relevant timesteps 
            transitions = corresponding_transitions[player]

            # Multiply in the rho corresponding to THIS player for each of the relevant timesteps! We query the behavior and evaluation policies correspondingly
            # The result is that each rho[t] is only multiplied into self._num_players times for simultaneous games or once for turn_based games
            for i, t in enumerate(transitions):
                if behavior_probs[i] == 0:
                    logging.error(" Detected a zero probability in the denominator for behavior probability. Will have a division by zero error. ")
                    raise Exception(" Rip. ")
                rho[t] *= evaluation_probs[i] / behavior_probs[i]
        return rho

    def get_policy_probabilities(self, trajectory_index, policy, query_player):
        """
        Get the policy probabilities of a particular trajectory and policy associated with a particular player
        There is additional cache functionality that stores the policy probabilities associated with policy_id, query_player, and trajectory_index

        trajectory_index: which trajectory's timesteps to query for 
        policy: which policy to query for probabilities
        query_player: the relevant player associated with the policy
        """

        # Check the cache. See if it is in there. If not, then query the policy 
        policy_id = policy.id
        # if policy_id in self._cache_policy_probabilities and query_player in self._cache_policy_probabilities[policy_id] and trajectory_index in self._cache_policy_probabilities[policy_id][query_player]:
        if len(self._cache_policy_probabilities[policy_id][query_player][trajectory_index]) > 0:
            info = self._cache_policy_probabilities[policy_id][query_player][trajectory_index]  # probs, relevant_timesteps
            return info[0], info[1]
        else: 
            corresponding_transitions = []

            query_info = {"info_state": [], "action": [], "legal_actions_mask": []}
            
            # For each of the transitions in trajectory (enumerate)
            trajectory = self._data[trajectory_index]
            for i, transition in enumerate(trajectory):
                # Get the relevant player 
                relevant_players = transition.relevant_players

                if query_player in relevant_players:
                    query_info["info_state"].append(transition.info_states[query_player])
                    query_info["action"].append([transition.actions[query_player]]) # Note, we insert [action] because we want the result to be a Nx1 size matrix
                    query_info["legal_actions_mask"].append(transition.legal_actions_masks[query_player])

                    # Insert the corresponding timestep of the query for the player
                    corresponding_transitions.append(i)
            
            # Query for each of the players 
            probs = policy.probabilities_with_actions(query_info["info_state"], query_info["action"], query_info["legal_actions_mask"], numpy=True)

            # Gets rid of the extra dimension turning matrix Nx1 to a vector of size N
            probs = np.reshape(probs, (-1))

            # The cache is a nested dictionary in the following order: policy -> player -> trajectory_index -> [probabilities, corresponding_transitions]
            self._cache_policy_probabilities[policy_id][query_player][trajectory_index] = [probs, corresponding_transitions]

            return probs, corresponding_transitions
            

    def apply_per_decision(self, all_rho_per_timestep):
        """
        Apply per-decision importance sampling. Given a list of length(trajectory), where each element is a list of individual rho values 
            indicating the singular likelihood ratio at each timestep, apply per-decision importance sampling by truncating each of the lists 
            only up to the current timestep...as future timesteps are irrelevant.

        all_rho_per_timestep: a list where each element is a list of singular liklihood ratios for each timestep
        """
        result = []
        for i, rho in enumerate(all_rho_per_timestep):
            relevant_rho = rho[: i+1]
            result.append(relevant_rho)
        return result

    def apply_weighted_likelihood_ratios(self, likelihood_ratios):
        """
        Apply weighted likelihood ratios. Given a list of lists, where each elements represents a single trajectory, containing the likelihood
            ratios to be applied to each reward within the trajectory, normalize all of these likelihood ratios depending on the sum of all likelihood 
            ratios across all trajectories. 
        """

        if not self._apply_per_decision:
            # This is special case where we actually don't want to scale our normalizer based on trajectory length
            normalizer = sum([lst[0] for lst in likelihood_ratios])
        else:
            raise NotImplementedError
        result = [[ratio / normalizer for ratio in lst] for lst in likelihood_ratios]
        return result

    #############################
    ## Importance Sampling End ##
    #############################

    #############################
    ## Statistics Calculation ###
    #############################

    def calculate_bayesian_correction(self, behavior_policies, source_evaluation_policies, target_evaluation_policies,
                                    source_evaluation_returns, source_is_evaluation_returns, target_evaluation_returns, player, indices_to_sample_from=[]):
        """
        Calculate the Bayesian correction term for target_evaluation-policies given source_evaluation_policies where the data was generated by 
        behavior_policies. 

        behavior_policies: a list of policy instances (one-per-player) that characterize the dataset
        source_evaluation_policies: a list of policy instances (one-per-player) in which we are transferring correction FROM
        target_evaluation-policies: a list of policy instances (one-per-player) in which we are transferring correction TO 
        source_evaluation_returns: Numpy array representing the "true" (lower variance) returns of source_evaluation_policies
        source_is_evaluation-returns: Numpy array representing the importance-sampled returns of source_evaluation_policies
        player: an integer indicating the player we would like to calculate bayesian correction for 
        indices_to_sample_from: the subset of the data we would like to use to calculate Bayesian correction
        """
        # Calculate the bayesian correction term for target_evauation_policies given source_evaluation_policies, the corresponding info,
        # behavior policies, and bootstrapping resampling parameters under some subset of the data indicated by 
        # indices_to_sample_from. The last parameter is used for distribution analyses, but is typically spans the 
        # entire dataset come game-solving.

        # You probably want to keep calculation of covariance and variance separate for independence 
        expected_returns = [source_evaluation_returns, target_evaluation_returns]
        difference_in_payoff_source = source_is_evaluation_returns - source_evaluation_returns

        # This correction is meant to be SUBTRACTED
        correction = self.calculate_p(behavior_policies, source_evaluation_policies, target_evaluation_policies, indices_to_sample_from, player, expected_returns) * difference_in_payoff_source[player]
        return correction

    def run_bootstrapped_samples(self, sample_size, num_estimates, estimator, data_indices_sample_from=[]):
        """
        Run bootstrapped sampling using the data.
            Sample Size: the number of samples per estimation 
            Num Estimates: the number of estimations to aggregate over 
            estimator: a lambda that takes in a list of length sample_size and outputs the metric
        Returns a list of length num_estimates of boostrapped estimates of the statistic
        """

        bootstrapped_samples = []
        # For num_estimates times: 
        time_last_update = time.time() 
        for iteration in range(num_estimates):
            # Sample sample_size indices from len(self._data)
            if len(data_indices_sample_from) > 0:
                indices = np.random.choice(data_indices_sample_from, sample_size)
            else:
                indices = np.random.randint(0, len(self._data), sample_size)

            if time.time() - time_last_update > 60:
                print("Progress of bootstrap samples: ", 100 * float(iteration) / num_estimates, " percent.")
                time_last_update = time.time()

            # Apply the estimator to the indices and record the results 
            result = estimator(indices)
            bootstrapped_samples.append(result)
        return bootstrapped_samples

    def calculate_p(self, behavior_policies, source_evaluation_policies, target_evaluation_policies, indices, player, expected_returns=[]):
        """
        Calculate the ratio between covariance and variance of IS estimates governed by source and target evaluation policies.

        behavior_policies: a list of policy instances (one-per-player) that characterize the dataset
        source_evaluation_policies: a list of policy instances (one-per-player) in which we are transferring correction FROM
        target_evaluation-policies: a list of policy instances (one-per-player) in which we are transferring correction TO 
        indices: indices of the dataset to use to calculate our p ratio
        player: the relevant player we are calculating the p ratio for
        expected_returns: possible "true" returns to inject into our calculation of the ratio P for lower variance (?) 
        """
        # expected_returns is a possibly populated list where index 0 is source and index 1 is target?
        # For each element, we have estimates of the expected returns for both players 

        # Evaluate policy using is_samples for each individual indices (that way we can calculate variance)
        source_returns = self.calculate_importance_sampled_evaluation(behavior_policies, source_evaluation_policies, indices=indices)[1]
        source_returns = [r[player] for r in source_returns]

        target_returns = self.calculate_importance_sampled_evaluation(behavior_policies, target_evaluation_policies, indices=indices)[1]
        target_returns = [r[player] for r in target_returns]


        source_returns = np.array(source_returns)
        target_returns = np.array(target_returns)

        # TODO: Insert if statement here to check if expected_returns has any useful information (?)
        # if len(expected_returns) == 0:
        source_sample_mean = np.mean(source_returns)
        target_sample_mean = np.mean(target_returns)
        # else:
        #     source_sample_mean = expected_returns[0][player]
        #     target_sample_mean = expected_returns[1][player]


        source_sample_variance = np.mean(np.square(source_returns - source_sample_mean))
        sample_covariance = np.mean(np.multiply(source_returns - source_sample_mean, target_returns - target_sample_mean))

        sample_p = sample_covariance / source_sample_variance 

        return sample_p


    ##############################
    # Statistics Calculation End #
    ##############################


    ##############################
    ###### Full Experiments ######
    ##############################
    def gaussian_plots_for_is_estimates(self, behavior_policies, evaluation_policies, sample_size, num_samples, plot_location, player):
        """
        This experiment analyzes the distribution of IS estimates under evaluation_policies 
        for a dataset (self._data) approximately generated by behavior_policies. The IS estimates
        will be of sample size sample_size. Here, we assume that behavior_policies is held fixed 
        across estimates...even though when we estimate these, even though these would vary for 
        each sample. 
        """
        # Take bootstrapped samples 
        is_estimator = lambda list_of_indices: self.calculate_importance_sampled_evaluation(behavior_policies, evaluation_policies, list_of_indices)[0]
        bootstrapped_samples = self.run_bootstrapped_samples(sample_size, num_estimates=num_samples, estimator=is_estimator)

        bootstrapped_samples = [sample[player] for sample in bootstrapped_samples]

        # distribution = np.random.normal(size=1000) this was a test
        number_of_bins = 30
        counts, bins = np.histogram(bootstrapped_samples, bins=number_of_bins)
        plt.stairs(counts, bins)
        plt.savefig(plot_location)

        logging.info("Plot generated at: " + plot_location)
        return 

    def visualize_covariance_between_is_estimates(self, behavior_policies, evaluation_policies_1, evaluation_policies_2, sample_size, num_samples, plot_location, player):
        """
        This experiment analyzes the joint distribution between IS estimates for two evaluation policies. 
        The purpose is to see covariance relationships between the two, depending on the degree to which 
        their trajectory-space coverages overlap. 
        """
        is_estimator = lambda list_of_indices: np.array([self.calculate_importance_sampled_evaluation(behavior_policies, evaluation_policies_1, list_of_indices)[0], 
                                                        self.calculate_importance_sampled_evaluation(behavior_policies, evaluation_policies_2, list_of_indices)[0]])

        # Each of these elements is going to be a np.array where the first element is evaluation_policies_1 and second is evaluation_policies_2
        # Then, for each of the elements, there are returns corresponding to each player
        bootstrapped_samples = self.run_bootstrapped_samples(sample_size, num_estimates=num_samples, estimator=is_estimator)

        evaluation_policy_1_returns = [sample[0][player] for sample in bootstrapped_samples]
        evaluation_policy_2_returns = [sample[1][player] for sample in bootstrapped_samples]

        plt.plot(evaluation_policy_1_returns, evaluation_policy_2_returns, 'bo')
        plt.savefig(plot_location)

        logging.info("Plot generated at: " + plot_location)
        return 

    def analyze_bayesian_correction(self, behavior_policies, source_evaluation_policies, target_evaluation_policies, sample_size, num_samples, plot_location, player, num_datapoints):
        """
        This is to analyze the distribution of the Bayesian correction term and the distribution change when 
        the correction is applied to a target policy's evaluation. These distributions will vary based on the 
        datapoints we sample from our dataset to hypothetically train our policies. 
        """
        # For num_datapoints 
        time_last_update = time.time()

        # Get the "true" return of the source policy 
        source_evaluation_returns = self.calculate_importance_sampled_evaluation(behavior_policies, source_evaluation_policies)[0]

        # Track the corrections 
        corrections = []
        target_is_returns = []
        target_corrected_returns = []

        for j in range(num_datapoints):
            if time.time() - time_last_update > 60: 
                print("Progress on the round of bootstrapping: ", 100 * float(j) / num_datapoints, " percent. ")
                time_last_update = time.time()
            # Decide which subset of the data (of size sample_size) to use 
            indices_of_current_sample = np.random.choice(len(self._data), size=sample_size)

            # Get the "importance-sampled" return of the source and target policies given the current indices we are calculating it for
            source_is_evaluation_returns = self.calculate_importance_sampled_evaluation(behavior_policies, source_evaluation_policies, indices_of_current_sample)[0]
            target_evaluation_returns = self.calculate_importance_sampled_evaluation(behavior_policies, target_evaluation_policies, indices_of_current_sample)[0]


            correction = self.calculate_bayesian_correction(behavior_policies, source_evaluation_policies, target_evaluation_policies,
                                    source_evaluation_returns, source_is_evaluation_returns, target_evaluation_returns, player, indices_of_current_sample)
            
            corrections.append(correction)
            target_is_returns.append(target_evaluation_returns[player])
            target_corrected_returns.append(target_evaluation_returns[player] - correction)

        target_evaluation_returns = self.calculate_importance_sampled_evaluation(behavior_policies, target_evaluation_policies)[0]

        # Compare the distributions between bayesian corrected and normal IS sampled evaluation
        number_of_bins = 30
        counts, bins = np.histogram(target_is_returns, bins=number_of_bins)
        plt.stairs(counts, bins)
        plt.savefig(plot_location+"_is")
        plt.clf() 

        counts, bins = np.histogram(target_corrected_returns, bins=number_of_bins)
        plt.stairs(counts, bins)
        plt.savefig(plot_location+"_corrected")

        print("")
        print(" Manual analysis: ")
        print(" Bias of IS samples: ", np.mean(np.array(target_is_returns) - target_evaluation_returns[player]), "   Bias of corrected IS samples: ", np.mean(np.array(target_corrected_returns) - target_evaluation_returns[player]))
        print(" Variance of IS samples: ", np.var(target_is_returns), "   Variance of corrected IS samples: ", np.var(target_corrected_returns))
        print("")
        return 

    # def analyze_covariance_variance_ratio(self, behavior_policies, source_evaluation_policies, target_evaluation_policies, sample_sizes, num_samples, plot_location, player):
    def analyze_covariance_variance_ratio(self, behavior_policies, source_evaluation_policies, target_evaluation_policies, sample_size, plot_location, player, num_datapoints):
        """
        Do analysis on various sample_sizes to see if there is a relationship between the covariance of two evaluation policies 
        and the variance of one of them. Num_samples is the bootstrapped estimation number for each of the covariance/variance 
        estimates and we choose to plot the relationship across sample_sizes for only a particular player.
        """

        p_estimator = lambda list_of_indices: self.calculate_p(behavior_policies, source_evaluation_policies, target_evaluation_policies, list_of_indices, player)
        
        all_p_estimates = self.run_bootstrapped_samples(sample_size, num_datapoints, p_estimator)

        # Look at the distribution of p_estimates (ratios of covariance to variance)
        number_of_bins = 30
        counts, bins = np.histogram(all_p_estimates, bins=number_of_bins)
        plt.stairs(counts, bins)
        plt.savefig(plot_location)
        plt.clf() 
        return 

    ##############################
    #### Full Experiments End ####
    ##############################