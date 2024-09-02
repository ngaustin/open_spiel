"""
This file is dedicated to being plugged into a policy wrapper. This can be trained using an offline dataset. Then, it is frozen after training. The "step" call is used from the policy wrapper 
to subsequently get the policy's action. 
"""
import numpy as np 
from absl import logging

from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.policy_wrapper import PolicyWrapper
from open_spiel.python.algorithms.offline_psro.utils.utils import compute_hash_string

class DiscreteBCPolicy(PolicyWrapper):


    def __init__(self, num_actions, state_size):
        super().__init__(self, num_actions, state_size) 

        self._state_counts = {} # maps a string representation of the state to a length num_actions list, representing the number of times an action was executed in the data

    def step(self, state, legal_actions):
        """
        state: current state/observation for the agent
        legal_actions: a list representing the indices of the possible actions at the given state/observation

        returns: index of the action chosen
        """
        hash_string = compute_hash_string(state)
        if hash_string in self._state_counts:
            action_to_counts = self._state_counts[hash_string]
            counts = [action_to_counts[a] for a in legal_actions]
            normalizer = float(sum(counts)())
            return np.random.choice(legal_actions, size=1, p=[c/normlalizer for c in counts])
        else:
            return np.random.choice(legal_actions)
    
    def train(self, data):
        """
        data: list representing the dataset in trajectory level units

        returns: a dictionary representing the training information such as training loss, q values, or any other metrics
        """
        for trajectory in data:
            for transition in trajectory:
                for p in transition.relevant_players:
                    if p in players:
                        info_state = transition.info_states[p]
                        action = transition.actions[p]
                        hash_string = compute_hash_string(info_state)
                        counts = self._state_counts.get(hash_string, [0 for _ in range(self._num_actions)])
                        counts[action] += 1
        return 

    def probabilities(self, state, legal_actions_mask, numpy=False):
        """
        state: a N x state_size matrix representing a bunch of states that we want to query for probabilities 
        legal_actions_mask: a N x num_actions mask where 1 means it is legal whereas 0 means it is illegal
        numpy: whether to return a numpy matrix instead of a tensorflow matrix (numpy is used for evaluation while tf can be used for gradients)

        returns: a N x num_actions matrix representing the probabilities of taking actions at each state
        """
        # Unfortunately, I don't know if there is a way to do this easily
        # Let's hope that these are all numpy arrays..otherwise this will be very inefficient
        all_probabilities = []
        for i in range(len(state.size()[0])):
            s = state[i]
            mask = legal_actions_mask[i] 
            hash_string = compute_hash_string(s)
            if hash_string in self._state_counts:
                action_to_counts = self._state_counts[hash_string]
                normalizer = sum(action_to_counts.values())
                probs = [l*action_to_counts.get(a) / normalizer for a, l in enumerate(mask)]
            else: 
                # Choose uniformly from the legal_actions_mask[]
                normalizer = np.sum(mask)
                probs = mask / normalizer 
            all_probabilities.append(probs)
        
        if numpy: 
            return np.array(all_probabilities)
        else:
            return tf.FloatTensor(all_probabilities)


    def probabilities_with_actions(self, state, action, legal_actions_mask, numpy=False):
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

