"""
This file is dedicated to wrapping any kind of policy to fit into the offline psro framework. This is used for both behavior policy 
generation and offline policy training (part of components A and B), which is why it is outside of the core folders. 
"""

from abc import ABC, abstractmethod

class PolicyWrapper(ABC):
    total_policies = 0

    def __init__(self, policy, num_actions, state_size):
        self._num_actions = num_actions
        self._state_size = state_size
        self._id = PolicyWrapper.total_policies
        PolicyWrapper.total_policies += 1

    @abstractmethod
    def step(self, step_object):
        raise NotImplementedError

    @property
    def num_actions(self):
        return self._num_actions
    
    @property 
    def state_size(self):
        return self._state_size

    @property
    def id(self):
        return self._id
        