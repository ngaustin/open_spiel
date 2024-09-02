""" 
File that brute force analyzes and combines files within the D_data folder
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

FLAGS = flags.FLAGS 

flags.DEFINE_string("experiment_name", "", "Experiment name")

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    if FLAGS.experiment_name == "compare_regret":
        names = ["1.0", "2.0", "5.0", "8.0","12.0", "15.0", "0.0"]
        directories = ["test_incentive_trials/parameter_(0)/", "test_incentive_trials/parameter_(1)/", "test_incentive_trials/parameter_(2)/", "test_incentive_trials/parameter_(3)/", 
        "test_incentive_trials/parameter_(4)/", "test_incentive_trials/parameter_(5)/", "test_incentive_trials/parameter_(6)/"]

        for experiment in directories:
            trials = [folder + "/" for folder in os.listdir(experiment) if not os.path.isfile(folder)]
            regret_aggregate = [[] for _ in range(30)]
            for trial in trials:
                file = [f for f in os.listdir(experiment+trial) if "true_game_regret_calculations" in f][0]

                with open(experiment + trial + "/" + file, 'rb') as f:
                    regret_values = pickle.load(f)

                for i, val in enumerate(regret_values[0]):
                    regret_aggregate[i].append(val)
                
            regret_aggregate = [sum(lst) / len(lst) for lst in regret_aggregate if len(lst) > 0]
            plt.plot(range(len(regret_aggregate)), regret_aggregate, label=names[int(experiment.split('_')[-1][1])])
            print("x: ", list(range(len(regret_aggregate))))
            print("Regret values: ", regret_aggregate)
        plt.title("Regret Player 0".format(experiment))
        plt.ylim(2.0, 3.6)
        plt.legend()
        plt.savefig('regret.jpg'.format(experiment))
        plt.clf()

    else:
        raise NotImplementedError
    print("Experiment done.")

if __name__ == "__main__":
  app.run(main)