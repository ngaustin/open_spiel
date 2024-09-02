"""
This file is dedicated to training a feed-forward world model
"""

import collections
import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import math 
import time

from abc import ABC, abstractmethod
from open_spiel.python import simple_nets

class WorldModelBase(ABC):
    def __init__(self, model, state_size, action_size, model_args={}):
        self._state_size = state_size
        self._action_size = action_size
        self._data = model_args["data"]
        self._session = model_args["session"]
        

    def get_next_state(self, state):
        raise NotImplementedError
    
    def train(self, num_gradient_steps):
        raise NotImplementedError 


