# This policy is used for proof of concept purposes. It is a fundamentally uniform random policy but takes in a parameter 
# alpha that skews the policy more and more towards a deterministic policy (where the policy chooses a single action at 
# each state but that single action is randomly determined before alpha is even considered).

"""
This file is dedicated to being plugged into a policy wrapper. This can be trained using an offline dataset. Then, it is frozen after training. The "step" call is used from the policy wrapper 
to subsequently get the policy's action. 
"""
import numpy as np 
from absl import logging
from copy import copy 

from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.policy_wrapper import PolicyWrapper
from open_spiel.python.algorithms.offline_psro.utils.utils import compute_hash_string

class DiscretePerturbedUniformRandomPolicy(PolicyWrapper):


    def __init__(self, num_actions, state_size, alpha=.8, seed=None):
        super().__init__(self, num_actions, state_size) 

        self._alpha = alpha # this is the additional probability weight the policy puts on chosen action. The rest is uniform.
        self._state_to_action = {}  # maps a hash string state to the chosen action it skews towards
        self._seed = seed


    def step(self, state, legal_actions):
        """
        state: current state/observation for the agent
        legal_actions: a list representing the indices of the possible actions at the given state/observation

        returns: index of the action chosen
        """
        hash_string = compute_hash_string(state)
        if hash_string in self._state_to_action:
            action = self._state_to_action 

            normalizer = len(legal_actions)
            mask = np.array([1 if a in legal_actions else 0 for a in range(self._num_actions)])

            uniform_probs = mask / normalizer 
            alpha_probs = np.zeros(len(mask))
            alpha_probs[action] += 1
        
            probs = self._alpha * alpha_probs + (1 - self._alpha) * uniform_probs
            probs = [probs[a] for a in legal_actions] 
            print("probs perturbed: ", probs)
            return np.random.choice(legal_actions, p=probs)
        else:
            print("Generating random choice. ")
            return np.random.choice(legal_actions)
    
    def train(self, data, players):
        """
        data: list representing the dataset in trajectory level units

        returns: a dictionary representing the training information such as training loss, q values, or any other metrics
        """
        if self._seed:
            logging.info("Seed set for discrete perturbed uniform random policy. ")
            np.random.seed(self._seed)
            
        for trajectory in data:
            for transition in trajectory:
                for p in transition.relevant_players:
                    if p in players:
                        info_state = transition.info_states[p]
                        action = transition.actions[p]
                        legal_actions_mask = transition.legal_actions_masks[p]
                        hash_string = compute_hash_string(info_state)
                        if hash_string not in self._state_to_action:
                            self._state_to_action[hash_string] = np.random.choice([action for action, is_legal in enumerate(legal_actions_mask) if is_legal])
        
        # print("Resulting state_to_action: ", self._state_to_action)

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
        for i in range(len(state)):
            s = state[i]
            mask = legal_actions_mask[i] 
            hash_string = compute_hash_string(s)
            
            normalizer = np.sum(mask)
            uniform_probs = mask / normalizer 
            if hash_string in self._state_to_action:
                action = self._state_to_action[hash_string]
                
                alpha_probs = np.zeros(len(mask))
                alpha_probs[action] += 1
            
                probs = self._alpha * alpha_probs + (1 - self._alpha) * uniform_probs 
            else: 
                # Choose uniformly from the legal_actions_mask[]
                probs = uniform_probs 
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

    
    def create_copy_with_noise(self, noise):
        """
        Creates a copy of this object but perturbs the alpha value slightly.
        Makes it so that a similar policy is created for analysis of the 
        relationship between trajectory coverage and covariance.

        Noise, in this case, is the amount of alpha to add from self._alpha. Note that noise can be negative 
        """

        new_alpha = np.clip(self._alpha + noise, 0, 1)
        copied_object = DiscretePerturbedUniformRandomPolicy(self._num_actions, self._state_size, new_alpha)

        copied_object._state_to_action = copy(self._state_to_action)
        return copied_object

