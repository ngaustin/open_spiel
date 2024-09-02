"""
This is dedicated to using a replicator dynamics adaptation to solve for profiles with minimal upper-bound regret in interval-based games.
"""

import collections
import time
from collections import defaultdict

import os
import matplotlib.pyplot as plt
from absl import logging
import numpy as np

class RDIntervalSolver:
    def __init__(self, steps=int(1e5), delta=1e-4):
        # For now, we assume we work with symmetric games
        self._num_steps = steps 
        self._delta = delta
        return 

    def solve_game_regret_upper_bound_optimization(self, interval_game, alpha=.5):
        # This solver greedily considers other players' deviation payoffs if the current player is to change their strategy. 
        # That way, we reduce the amount of upper bound regret a player could incur on another
        
        num_strategies = interval_game.shape[2]
        profile = np.array([1.0 / num_strategies for _ in range(num_strategies)])
        upper_bound_regrets = []
        
        # TODO: Create the deviation payoff matrix for easier processing 
        deviation_payoff_matrix = np.zeros((2, num_strategies, num_strategies))
        for player in range(len(interval_game)):
            if player == 0:
                # Get player 1's lower and upper bounds 
                lower_bound = interval_game[1][0]
                upper_bound = interval_game[1][1]

                # Make an S x S matrix where each entry is the maximum regret given the current lower bound 
                for i in range(num_strategies):
                    for j in range(num_strategies):
                        regret = np.max(upper_bound[i]) - lower_bound[i][j]
                        deviation_payoff_matrix[0][i][j] = regret 
            elif player == 1:
                # Get player 0's lower and upper  bounds 
                lower_bound = interval_game[0][0]
                upper_bound = interval_game[0][1]

                # Make an S x S matrix where each entry is hte maximum regret given the current lower bound 
                for i in range(num_strategies):
                    for j in range(num_strategies):
                        regret = np.max(upper_bound[:, j]) - lower_bound[i][j]
                        deviation_payoff_matrix[1][i][j] = regret 
            else:
                raise NotImplementedError
        # For a bunch of steps 
        # print("Deviation payoff matrix: ", deviation_payoff_matrix[0])
        for i in range(self._num_steps):
            profile, upper_bound_regret = self.rd_upper_bound_step(interval_game, deviation_payoff_matrix, profile, alpha)

            upper_bound_regrets.append(np.sum(upper_bound_regret))

        # print("Output profile: ", profile)
        return profile, upper_bound_regrets 

    def solve_game(self, interval_game):
        # This solver greedily does RD steps that put more weight on strategies that have a low regret upper bound given the current profile.

        # interval_game is of shape 2 x 2 x S x S 
        # N is the number of players 
        # 2 represents the lower and upper bounds 
        # S x S is the size of the normal form game 

        num_strategies = interval_game.shape[2]
        profile = np.array([1.0 / num_strategies for _ in range(num_strategies)])
        upper_bound_regrets = []
        # For a bunch of steps 
        for i in range(self._num_steps):
            # Append the regret values and modify current profile 
            # print("Iteration: ", i)
            profile, upper_bound_regret = self.rd_step(interval_game, profile)
            print("upper  bound regret: ", upper_bound_regret[0])
            print("profile: ", profile)
            print('')
            # if i > 2 and np.sum(upper_bound_regret) > upper_bound_regrets[-1]:
            #     print("stopped at : ", i)
            #     print('profile: ', profile)
            #     break

            upper_bound_regrets.append(np.sum(upper_bound_regret))

            
        # print("regrets: ", upper_bound_regrets)
        return profile, upper_bound_regrets

    def rd_step(self, interval_game, profile):
        # interval_game is of shape N x 2 x S x S 
        # N is the number of players 
        # 2 represents the lower and upper bounds 
        # S x S is the size of the normal form game 

        # Calculate the upper bound deviation payoff
        lower_bound_deviation_payoff, upper_bound_deviation_payoff, lower_bound_current_payoff,  upper_bound_current_payoff = self.calculate_bounds(interval_game, profile)  # 2 x S, 2

        # Calculate upper bound regret 
        upper_bound_regret = np.max(upper_bound_deviation_payoff, axis=1) - lower_bound_current_payoff

        # Instead of using deviation payoffs to update profile, we will calculate the upper bound regret IF players were to deviate to a different strategy given the current profile
        upper_bound_regret_if_deviate = np.expand_dims(upper_bound_current_payoff, axis=1) - lower_bound_deviation_payoff # 2 x S
        upper_bound_regret_if_deviate = np.expand_dims(np.max(upper_bound_deviation_payoff, axis=1), axis=1) - lower_bound_deviation_payoff

        # The higher the number, the worse it is. So, we subtract. 
        new_profile = profile - (self._delta) * profile * upper_bound_regret_if_deviate
        new_profile = new_profile / np.expand_dims(np.sum(new_profile, axis=1), axis=1)
        new_profile = new_profile[0]

        print("BOUNDS: ", lower_bound_deviation_payoff[0], upper_bound_deviation_payoff[0], lower_bound_current_payoff[0],  upper_bound_current_payoff[0])
        print("DEVIATION REGRET: ", upper_bound_regret_if_deviate[0])

        # print("LB DP:", lower_bound_deviation_payoff, "   UB DP: ", upper_bound_deviation_payoff, "   LB CP: ", lower_bound_current_payoff, "   UB CP: ", upper_bound_current_payoff)

        # Return the upper bound regret and new profile
        # new_profile = profile + (self._delta) * profile * upper_bound_deviation_payoff
        # new_profile = new_profile / np.expand_dims(np.sum(new_profile, axis=1), axis=1)
        # print("new profiles: ", new_profile)
        # new_profile = new_profile[0]

        return new_profile, upper_bound_regret

    def rd_upper_bound_step(self, interval_game, deviation_payoff_matrix, profile, alpha):
        _, upper_bound_deviation_payoff, lower_bound_current_payoff, _ = self.calculate_bounds(interval_game, profile)  # 2 x S, 2

        # Calculate upper bound regret 
        upper_bound_regret = np.max(upper_bound_deviation_payoff, axis=1) - lower_bound_current_payoff
        new_profile = []

        # # TODO: Use the deviation payoff matrix to get the new profiles using the OTHER player's profiles 
        # for player in range(len(interval_game)):
        #     # For each player, indexing deviation_payoff_matrix will give the other player's regrets of they were to play the corresponding strategy
        #     curr_regrets = deviation_payoff_matrix[player]

        #     expected_regrets = []
        #     num_strategies = interval_game.shape[2]
        #     for i in range(num_strategies):
        #         if player == 0:
        #             # The regrets should be calculated using the columns...so index the row.
        #             regrets_for_other_player = curr_regrets[i, :]
        #             expected_regret_for_other_player = np.sum(profile * regrets_for_other_player)
        #             expected_regrets.append(expected_regret_for_other_player)
        #         if player == 1:
        #             # In this case, we calculate using the rows...so index the column 
        #             regrets_for_other_player = curr_regrets[:, i]
        #             expected_regret_for_other_player = np.sum(profile * regrets_for_other_player)
        #             expected_regrets.append(expected_regret_for_other_player)
            
        #     # Expected_regrets should be length S vector corresponding to, if I were to play a certain strategy, how much regret potential is there for the OTHER player given the current profile 
        #     # So, we change the player's profile by minimizing this. Higher regrets means less weight put in that strategy 
        #     curr_new_profile = profile - (self._delta) * profile * np.array(expected_regrets)
        #     curr_new_profile = curr_new_profile / np.sum(curr_new_profile)
        #     new_profile.append(curr_new_profile)

        # TODO: Use negated version of the deviation_payoff_matrix
        # Do one classic RD_step (without any fancy stuff!) on the negated matrix 
        negated_regret_matrix = -1 * deviation_payoff_matrix[0]
        # Assume symmetry
        deviation_regrets = (alpha * negated_regret_matrix + interval_game[0][0] * (1 - alpha)) @ profile.T
        # print(profile.shape, deviation_regrets.shape)
        new_profile = profile + (self._delta) * profile * deviation_regrets
        new_profile = new_profile / np.sum(new_profile)
        # print(new_profile.shape)

        # print("new profile: ", new_profile)
        
        return new_profile, upper_bound_regret

    def calculate_bounds(self, interval_game, profile):
        # interval_game is of shape 2 x 2 x S x S 
        # N is the number of players 
        # 2 represents the lower and upper bounds 
        # S x S is the size of the normal form game 

        # Profile is of shape S (symmetric assumed)


        # Calculate the expected lower bound payoff 
        lower_bound_matrices = interval_game[:, 0, :, :]  # 2 x S x S
        upper_bound_matrices = interval_game[:, 1, :, :]  # 2 x S x S

        outer_product_matrix = np.outer(profile, profile) # S x S
        outer_product_matrix = np.expand_dims(outer_product_matrix, axis=0) # 1 x S x S
        repeated_matrix = np.repeat(outer_product_matrix, upper_bound_matrices.shape[0], axis=0)  # 2 x S x S

        expected_upper = upper_bound_matrices * repeated_matrix  # 2 x S x S
        upper_bound_current_payoff = np.sum(np.sum(expected_upper, axis=2), axis=1)  # 2
        # print("expected_lower: ", expected_lower)

        expected_lower = lower_bound_matrices * repeated_matrix # 2 x S x S
        lower_bound_current_payoff = np.sum(np.sum(expected_lower, axis=1), axis=1) # 2

        # Calculate the upper bound deviation payoff 
        

        # For player 0, we need to multiply profile into the columns 
        deviation_payoffs_lower_0 = lower_bound_matrices[0] @ profile.T
        deviation_payoffs_upper_0 = upper_bound_matrices[0] @ profile.T

        # For player 1, we need to multiply profile into the rows 
        deviation_payoffs_lower_1 = profile @ lower_bound_matrices[1]
        deviation_payoffs_upper_1 = profile @ upper_bound_matrices[1]

        
        lower_bound_deviation_payoffs = np.vstack((deviation_payoffs_lower_0, deviation_payoffs_lower_1))  # 2 x S
        upper_bound_deviation_payoffs = np.vstack((deviation_payoffs_upper_0, deviation_payoffs_upper_1)) # 2 x S

        return lower_bound_deviation_payoffs, upper_bound_deviation_payoffs, lower_bound_current_payoff, upper_bound_current_payoff
         
    