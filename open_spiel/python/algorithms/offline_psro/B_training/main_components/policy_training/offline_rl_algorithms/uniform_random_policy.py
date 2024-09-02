"""
This file is dedicated to being plugged into a policy wrapper. This can be trained using an offline dataset. Then, it is frozen after training. The "step" call is used from the policy wrapper 
to subsequently get the policy's action. 
"""
import numpy as np 
from absl import logging

from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.policy_wrapper import PolicyWrapper

class UniformRandomPolicy(PolicyWrapper):


    def __init__(self, num_actions, state_size):
        super().__init__(self, num_actions, state_size) 

    def step(self, step_object, player, session, is_evaluation=False):
        """

        returns: index of the action chosen
        """
        if step_object.is_terminal: 
            return None, []
        
        legal_actions_mask = np.array(step_object.legal_actions_mask)
        probs = legal_actions_mask/np.sum(legal_actions_mask)
        return np.random.choice(range(len(step_object.legal_actions_mask)), p=probs), probs
    
    def get_graph(self):
        return None
    
    def train(self, data):
        """
        data: list representing the dataset in whatever unit (transition level or trajectory level)

        returns: a dictionary representing the training information such as training loss, q values, or any other metrics
        """
        logging.info("No training necessary for uniformly random policy. Bypassing. ")
        pass 

    def probabilities(self, step_object, numpy=False):
        """
        state: a N x state_size matrix representing a bunch of states that we want to query for probabilities 
        legal_actions_mask: a N x num_actions mask where 1 means it is legal whereas 0 means it is illegal
        numpy: whether to return a numpy matrix instead of a tensorflow matrix (numpy is used for evaluation while tf can be used for gradients)

        returns: a N x num_actions matrix representing the probabilities of taking actions at each state
        """
        if numpy: 
            normalizer = np.reshape(np.sum(legal_actions_mask, axis=1), (-1, 1))
            return legal_actions_mask / normalizer
        else: 
            # TODO: This might be wrong because of array shape issues
            normalizer = tf.reduce_sum(legal_actions_mask, axis=1).reshape((-1, 1))
            return tf.FloatTensor(legal_actions_mask) / normalizer

    def probabilities_with_actions(self, step_object, numpy=False):
        """
        state: a N x state_size matrix representing a bunch of states that we want to query for probabilities 
        action: a N x 1 matrix representing the actions we want to query for each of the states 
        legal_actions_mask: a N x num_actions mask where 1 means it is legal whereas 0 means it is illegal
        numpy: whether to return a numpy matrix instead of a tensorflow matrix (numpy is used for evaluation while tf can be used for gradients)

        returns: a N x 1 matrix representing the probabilities of taking a particular action at each state
        """
        probabilities = self.probabilities(state, legal_actions_mask, numpy)

        if numpy: 
            return np.take_along_axis(probabilities, np.array(action), axis=1)
        else: 
            return tf.gather(probabilities, action, axis=1, batch_dims=1)

