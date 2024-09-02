
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
from open_spiel.python import rl_environment
from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.world_model_deterministic import WorldModelDeterministic

FLAGS = flags.FLAGS

# Dataset choice 
flags.DEFINE_string("experiment_name", "test_model_training", "Name of the experiment we are running")
flags.DEFINE_string("dataset_path", "", "Path to file to load the offline dataset")
flags.DEFINE_string("game_name", "", "Name of the game we are analyzing")

# Model Training Related 
flags.DEFINE_integer("model_width", 50, "Width of the model")
flags.DEFINE_integer("model_depth", 2, "Number of hidden layers in model")
flags.DEFINE_integer("ensemble_size", 5, "Number of models in the ensemble")
flags.DEFINE_integer("batch_size", 32, "Batch size of learning dynamics model")
flags.DEFINE_float("learning_rate", 3e-4, "Learning rate for model training") 
flags.DEFINE_float("halt_threshold", .1, "Maximum prediction difference for state-action pair to be considered OOD")

# Plot-Related

# Misc
flags.DEFINE_integer("seed", 1, "Seed for random")



def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print("\n\n")
    np.random.seed(FLAGS.seed)

    with open(FLAGS.dataset_path, "rb") as npy_file:
        data = np.load(npy_file, allow_pickle=True)
    concatenated_data = []
    for rollout in data:
        concatenated_data.extend(rollout)
    data = concatenated_data

    pyspiel_game = pyspiel.load_game(FLAGS.game_name)
    logging.info("Loaded game: %s", FLAGS.game_name)

    # Use rl_environment as a wrapper and specify the observation type
    env = rl_environment.Environment(pyspiel_game, observation_type=rl_environment.ObservationType.OBSERVATION)
    state_size = len(data[0].global_state)
    action_size = 1 if env.is_turn_based else env.num_players

    if FLAGS.experiment_name == "test_model_training":
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)

    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
  
    with tf.Session(config=session_conf) as sess:

        model_args = {"hidden_sizes": [FLAGS.model_width] * FLAGS.model_depth,
                      "batch_size": FLAGS.batch_size,
                      "learning_rate": FLAGS.learning_rate,
                      "ensemble_size": FLAGS.ensemble_size,
                      "halt_threshold": FLAGS.halt_threshold, 
                      "observation_sizes":[env.observation_spec()["info_state"][0]] * env.num_players,
                      "num_players":env.num_players,
                      "num_actions":env.action_spec()["num_actions"],
                      "data": data, 
                      "session": sess}

        model = WorldModelDeterministic(state_size, action_size, model_args)
        losses = model.train(num_gradient_steps=10000)
        print("Losses: ", losses)

        plt.plot(range(len(losses)), losses)
        plt.savefig('model_loss.jpg')
    
    print("Finished experiment.")
    


if __name__ == "__main__":
  app.run(main)
