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
        names = ["RRD, Lambda = 2, Alpha = .3",
                 "RRD, Lambda = 4, Alpha = .3",
                 "RRD, Lambda = 6, Alpha = .3",
                 "R3D, Lambda = 2, Alpha = .3",
                 "R3D, Lambda = 4, Alpha = .3",
                 "R3D, Lambda = 6, Alpha = .3",
                ]
        # names = ["eps = 0.0",
        #         "eps = 0.1",
        #         "eps = 0.2"]
        directories = [ ["tuning_main_experiment/parameter_(6)/"],
                        ["tuning_main_experiment/parameter_(7)/"],
                        ["tuning_main_experiment/parameter_(8)/"],
                        ["tuning_main_experiment/parameter_(15)/"],
                        ["tuning_main_experiment/parameter_(16)/"],
                        ["tuning_main_experiment/parameter_(17)/"]
                    ]
        graph_indices = [0, 1, 2, 3, 4, 5]

        for graph_index in graph_indices:
            regret_aggregate = [[] for _ in range(50)]
            for experiment in directories[graph_index]:
                trials = [folder + "/" for folder in os.listdir(experiment) if not os.path.isfile(folder)]
                

                for trial in trials:
                    print("Folder: ", experiment+trial)
                    file = [f for f in os.listdir(experiment+trial) if "true_game_regret_calculations" in f][0]

                    with open(experiment + trial + "/" + file, 'rb') as f:
                        regret_values = pickle.load(f)
                    print("Curr: ", regret_values)

                    for i, val in enumerate(regret_values[0]):
                        regret_aggregate[i].append(val)
                    
                    # plt.plot(range(len(regret_values[0])), regret_values[0])
                    
                    # local_save_path = '../graphs/' + experiment + trial
                    # if not os.path.exists(local_save_path):
                    #     os.makedirs(local_save_path)
                    # plt.savefig(local_save_path + 'regret.jpg')
                    # plt.clf()
            
            regret_means = [sum(lst) / len(lst) for lst in regret_aggregate if len(lst) > 0]
            regret_stds = np.array([np.std(lst) for lst in regret_aggregate if len(lst) > 0])
            plt.fill_between(range(len(regret_stds)), np.array(regret_means) - regret_stds, np.array(regret_means) + regret_stds, alpha=0.2)
            plt.plot(range(len(regret_means)), regret_means, label=names[graph_index])
            print("x: ", list(range(len(regret_means))))
            print("Regret values: ", regret_means)
            print("Minimum regret: ", np.min(regret_means))
        plt.title("Regret Player 0".format(experiment))
        plt.legend()
        plt.savefig('regret.jpg'.format(experiment))
        plt.clf()

    else:
        raise NotImplementedError
    print("Experiment done.")

if __name__ == "__main__":
  app.run(main)