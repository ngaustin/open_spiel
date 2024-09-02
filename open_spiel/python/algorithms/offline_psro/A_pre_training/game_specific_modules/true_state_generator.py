"""
This file is dedicated to wrapping any kind of true state generator for simulation purposes.
"""

from abc import ABC, abstractmethod

class TrueStateGeneratorWrapper(ABC):

    def __init__(self, game_name):
        self._game_name = game_name

    @abstractmethod
    def to_true_state(self, info, pyspiel_state):
        # This will vary heavily by game. Info is a dictionary full of stuff
        raise NotImplementedError

    @property 
    def true_state_size(self):
        return self._true_state_size
