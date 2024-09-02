"""
This module is dedicated to saving and loading tensorflow models with individual sessions. That way, when we construct multiple 
target samplers or RL policies of the same object type, we do not run into issues of colliding variable names in Tensorflow.

Furthermore, we provide functionality to freeze policies for faster inference and easier imports.

"""



import time
import collections
import numpy as np
import os
import pyspiel
import tensorflow.compat.v1 as tf
import sys

import matplotlib.pyplot as plt

from datetime import datetime
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

class TFModelManagement:
    def __init__(self):
        return 

    
    def freeze_graph(self, model_dir, name_frozen_model, output_node_names):
        """
        Given a model_dir of a saved data, we will load the model and subsequently save a frozen version of the model for later inference.
        NOTE: Requires that we are not currently in a Session (as we define a new session below)
        """

        checkpoint = tf.train.get_checkpoint_state(model_dir)
        input_checkpoint = checkpoint.model_checkpoint_path

        absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
        output_graph = absolute_model_dir + "/" + name_frozen_model

        clear_devices = True
        with tf.Session(graph=tf.Graph()) as sess:
            saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
            saver.restore(sess, input_checkpoint)

            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                tf.get_default_graph().as_graph_def(),
                output_node_names
            )

            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            # print("{} ops in the final graph.".format(len(output_graph_def.node)))
        print('\n')

    
    def load_frozen_graph(self, save_path, frozen_graph_name):
        """
        Returns a frozen graph for future inference. Note that this requires we do not have an active session currently.
        """

        graph_file_name = save_path + frozen_graph_name
        with tf.gfile.GFile(graph_file_name, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        return graph


    def save(self, session, name):
        """
        Given a session, save the currently defined TF variables into the checkpoint
        """

        saver = tf.train.Saver()
        if '.ckpt' not in name:
            logging.error("Improper form for tensorflow model save name.")
            raise Exception 
        saver.save(session, name)
        return 
    
