"""
This file is dedicated to created a Bargaining uniform action policy that is "somewhat" reasonable. It will enforce two things:

1. Given the opponent's previous offer, a player will only consider counter-offers that are strictly better for themself
2. We provide a hyperparameter that indicates the minimum probability of accepting an offer (set to relatively high)
   The reason for this is that sampling uniformly from all actions makes it so that the probability of accepting an offer
   very low...meaning all trajectories in our dataset contain very little deals being made.
"""
import numpy as np 
from absl import logging
import copy

from open_spiel.python.algorithms.offline_psro.B_training.main_components.policy_training.offline_rl_algorithms.policy_wrapper import PolicyWrapper

class BargainingUniformRandomPolicy(PolicyWrapper):


    def __init__(self, bargaining_true_state_generator, minimum_acceptance_probability, pyspiel_game, num_actions, state_size):
        super().__init__(self, num_actions, state_size) 
        self._true_state_generator = bargaining_true_state_generator
        self._minimum_acceptance_probability = minimum_acceptance_probability
        self._pyspiel_game = pyspiel_game

        # self._all_offers = []
        # for a in range(self._pyspiel_game.num_distinct_actions() - 1): # Accept action is the last action
        #     a_string = self._pyspiel_game.action_to_string(0, a)
        #     offer = a_string.split(' ')
        #     self._all_offers.append([int(num) for num in offer[-3:]])
        
        # print("All offers: ", self._all_offers)

    def step(self, step_object, player, session, is_evaluation=False):
        """

        returns: index of the action chosen
        """
        if step_object.is_terminal: 
            return None, []

        # We generate the dataset using observations and reconstruct the info_state later in our pipeline. info_state is a misnomer
        observation = step_object.info_state

        index_start_pool = self._true_state_generator.index_start_pool
        index_start_values = self._true_state_generator.index_start_values
        index_start_offers = self._true_state_generator.index_start_offers

        pool_vector = observation[index_start_pool: index_start_values]
        value_vector = observation[index_start_values: index_start_offers]
        previous_offer_vector = observation[index_start_offers:]

        # Convert each of these vectors into vectors with integers instead of indicators
        pool_items = self.vector_to_item_integers(pool_vector, self._true_state_generator.num_item_types)
        value_items = self.vector_to_item_integers(value_vector, self._true_state_generator.num_item_types)
        if sum(previous_offer_vector) > 0:
            # This means there was an offer last turn (it is NOT the first offer being made this game)
            previous_offer = self.vector_to_item_integers(previous_offer_vector, self._true_state_generator.num_item_types)

            # Calculate how much utility this is worth to us
            remaining_items = [pool_count - offer_count for pool_count, offer_count in zip(pool_items, previous_offer)]
            previous_utility = sum([value * remaining_count for value, remaining_count in zip(value_items, remaining_items)])

        else:
            # First offer being made this game
            previous_utility = 0
       
        legal_actions_mask = step_object.legal_actions_mask
        options = copy.copy(legal_actions_mask)

        # Gotten rid of this to enforce uniform action 
        
        # for action, is_legal in enumerate(legal_actions_mask):
        #     if is_legal and action != len(self._all_offers):  # legal and is not the accept action. Accept is always legal
        #         offer = self._all_offers[action]
        #         new_utility = sum([value * count for value, count in zip(value_items, offer)])
        #         if new_utility <= previous_utility: 
        #             options[action] = 0
        
        options = np.array(options)
        # Use our options as our probs instead of legal_actions_mask
        if 1.0 / sum(options) < self._minimum_acceptance_probability and legal_actions_mask[-1] == 1:
            prob_for_non_accept_actions = (1 - self._minimum_acceptance_probability) / (sum(options) - 1)
            probs = options * prob_for_non_accept_actions
            probs[-1] = self._minimum_acceptance_probability
        else:
            probs = options/np.sum(options)

        # Edge case 
        if np.isnan(probs).any():
            probs = legal_actions_mask / np.sum(legal_actions_mask)

        return np.random.choice(range(len(step_object.legal_actions_mask)), p=probs), probs
    
    def vector_to_item_integers(self, vector, num_item_types):
        """
        Any indicator vector (pool, value, offers) contains information regarding the various item types. If we split the vector into num_item_types pieces,
        each piece describes how many/how valuable each item is (depending on the context i.e. pool, value, or offer vector). This value can take on a variety
        of values 0-[some kind of max]. Hence, each piece describing one item should be of length [some kind of max] + 1 to include 0 as an option.
        Hence, the value of interest will be sum(vector_piece) - 1. We return one of these values for each item type.
        """
        total_elements = len(vector)
        assert total_elements % num_item_types == 0
        vector_subset_length = int(total_elements / num_item_types)

        integer_vector = []
        for i in range(num_item_types):
            value = sum(vector[i*vector_subset_length: (i+1)*vector_subset_length]) - 1
            integer_vector.append(value)

        return integer_vector

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

