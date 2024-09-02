
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
from open_spiel.python.algorithms.offline_psro.B_training.main_components.mss.minimax_solver import RDIntervalSolver

FLAGS = flags.FLAGS

# Dataset choice 
flags.DEFINE_string("experiment_name", "test_rd_interval_solver", "Name of the experiment we are running")

# Plot-Related


# Misc
flags.DEFINE_integer("seed", 1, "Seed for random")

# This is a symmetric game
interval_game = [[[[0, 1, 0], [-1, 1, -2], [-2, -3, -1]],  # Player 1 lower bound  
                  [[3, 4, 3], [ 1, 2,  0], [ 4,  2,  7]]], # Player 1 upper bound

                 [[[0, -1, -2], [1, 1, -3], [0, -2, -1]], # Player 2 lower bound 
                  [[3,  1,  4], [4, 2,  2], [3,  0,  7]]]] # Player 2 upper bound

interval_game = np.array(interval_game)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print("\n\n")
    np.random.seed(FLAGS.seed)

    if FLAGS.experiment_name == "test_rd_interval_solver":
        solver = RDIntervalSolver()
        profile, regret = solver.solve_game_regret_upper_bound_optimization(interval_game)
        print("Output profile: ", profile)
        plt.plot(range(len(regret)), regret)
        plt.savefig('test.jpg')
    if FLAGS.experiment_name == "test_rd_interval_solver_multiple_games":
        solver = RDIntervalSolver()
        num_games = 10
        num_strategies = 10
        games = []
        # Generate a bunch of random games. 
        # Create the lower and upper bounds for player 1 
        for i in range(num_games):
            lower_bound = (np.random.rand(num_strategies, num_strategies) * 10) - 5
            delta = (np.random.rand(num_strategies, num_strategies) * 5)
            # for j in range(num_strategies - 1):
            #     delta[j, j+1] = 0
            #     delta[j+1, j] = 0
            # delta[0, 0] = 0

            upper_bound = lower_bound + delta

            game = np.array([[lower_bound, upper_bound], [lower_bound.T, upper_bound.T]])
            games.append(game)

        print("Generated all games")

        for i, game in enumerate(games):
            print("Current game: ", game)
            for j, curr_alpha in enumerate([0, .2, .4, .6, .8, 1.0]):
                profile, regret = solver.solve_game_regret_upper_bound_optimization(game, curr_alpha)
                plt.plot(range(len(regret)), regret)
                plt.savefig('minimax_regrets/regret_{}_alpha_{}'.format(i, j))
                plt.clf()
                print("Saved regret iteration: ", i)
    print("Finished solving game")
    


if __name__ == "__main__":
  app.run(main)
