


from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.true_state_generator import TrueStateGeneratorWrapper
import numpy as np
import copy 

class BargainingTrueStateGenerator(TrueStateGeneratorWrapper):

    def __init__(self, game_name, info_state_size):
        super().__init__(game_name)
        self._info_state_size = info_state_size

    def set_information(self, info):
        self._info = info
        print("Info: ", self._info)

    def to_true_state(self, observations, pyspiel_state):
        assert len(observations[0]) > self._info["index_start_offers"]
        curr = observations[0]  # we include the most recent offer because that determines the return distribution if the next action is accept
        curr = curr + observations[1][self._info["index_start_values"] : self._info["index_start_offers"]]
        curr = curr + [pyspiel_state.current_player()]
        return curr

    def project_to_true_initial_state(self, noisy_initial_state):
        """
        Takes a noisy_initial_state and projects it into the space of "valid" initial states in the game. 
        This validity is not a harsh assumption; we just want representations that make physical sense.
        For example, it makes no sense to start in a state with an agreement already made, having a non-zero number of 
        offers already made, or a player having two different valuations for the same item.  """

        ################## Agreement Reached and Turns Passed ##################
        noisy_initial_state_copy = np.copy(noisy_initial_state)

        # The first value is an indicator of whether an agreement has been reached. This is actually set to 1.0 initially.
        offset = 0
        noisy_initial_state_copy[offset] = 1.0

        offset += 1
        # The next 10 values indicate how many offers have been made. These should all be 0.
        noisy_initial_state_copy[offset:self._max_turns+1] = 0.0
        offset += self._max_turns + 1
        ################ Agreement Reached and Turns Passed End ################

        ################ Pool Description Validity Check ################
        # There can be only be one number describing each item (Ex: you can't have both 2 and 3 basketballs at a given time)
        # The number of items of a particular type is described by the number of ones in sequence minus 1 at the beginning.
        # Minus 1 is to account for the possibility of having 0 counts of an item
        # [1.0 1.0 1.0 0.0 0.0] indicates there are 2 of an item and there is a max of 4 possible in any pool.
        for i in range(self._num_item_types):
            # Take the current count vector for a particular item
            curr_count_vector = noisy_initial_state[offset: offset + self._pool_max_num_items + 1]

            # Check which vector it is most similar of the self._pool_max_num_items + 1 vectors
            lowest_mse, most_similar_vector = np.inf, None
            curr_candidate = np.zeros(self._pool_max_num_items+1)
            for j in range(self._pool_max_num_items + 1):
                curr_candidate[j] = 1.0
                curr_mse = np.mean(np.square(curr_count_vector - curr_candidate))
                if curr_mse < lowest_mse:
                    lowest_mse = curr_mse
                    most_similar_vector = np.copy(curr_candidate)

            # Replace the current count vector with that vector
            noisy_initial_state_copy[offset: offset + self._pool_max_num_items + 1] = most_similar_vector

            # Look at the next count vector offset
            offset += self._pool_max_num_items + 1
        ############## Pool Description Validity Check End ##############
    
        ############## Player 1 Valuation Validity Check ##############
        # Player 1 Valuations are organized the same way but the offset is by self._total_value_all_items 
        for i in range(self._num_item_types):
            curr_value_vector = noisy_initial_state[offset: offset + self._total_value_all_items + 1]

            # Check which vector it is most similar of the self._pool_max_num_items + 1 vectors
            lowest_mse, most_similar_vector = np.inf, None
            curr_candidate = np.zeros(self._total_value_all_items+1)
            for j in range(self._total_value_all_items + 1):
                curr_candidate[j] = 1.0
                curr_mse = np.mean(np.square(curr_value_vector - curr_candidate))
                if curr_mse < lowest_mse:
                    lowest_mse = curr_mse
                    most_similar_vector = np.copy(curr_candidate)

            noisy_initial_state_copy[offset: offset+self._total_value_all_items+1] = most_similar_vector
            offset += self._total_value_all_items+1
        ############ Player 1 Valuation Validity Check End ############
        
        ############ Recent Offer Validity Check ############
        # Next is the most recent offer description. Since there was no offer at the beginning, these should all be zeros.
        noisy_initial_state_copy[offset: offset + self._info["length_each_offer"]] = 0
        offset += self._info["length_each_offer"]
        ########## Recent Offer Validity Check End ##########

        ########## Other Player Valuation Validity Check ##########
        for i in range(self._num_item_types):
            curr_value_vector = noisy_initial_state[offset: offset + self._total_value_all_items + 1]

            # Check which vector it is most similar of the self._pool_max_num_items + 1 vectors
            lowest_mse, most_similar_vector = np.inf, None
            curr_candidate = np.zeros(self._total_value_all_items+1)
            for j in range(self._total_value_all_items + 1):
                curr_candidate[j] = 1.0
                curr_mse = np.mean(np.square(curr_value_vector - curr_candidate))
                if curr_mse < lowest_mse:
                    lowest_mse = curr_mse
                    most_similar_vector = np.copy(curr_candidate)

            noisy_initial_state_copy[offset: offset+self._total_value_all_items+1] = most_similar_vector
            offset += self._total_value_all_items+1
        ########## Other Player Valuation Validity Check ##########
        
        ########## Player Turn Validity Check ##########
        # Finally, look at the player starting. This can be either of them. 
        noisy_initial_state_copy[-1] = np.clip(noisy_initial_state[-1], 0, 1)
        starting_player = noisy_initial_state_copy[-1]
        ######## Player Turn Validity Check End ########

        return noisy_initial_state_copy, [int(starting_player)]


    def true_state_symmetric_permute(self, true_state, permutation):
        # Converts a true state to the corresponding permutation. Permutation is a length num_players list 
        # where permutation[i] corresponds to player i becoming player permutation[i] instead.

        # Note: returns depend on the player that made the offer (because the bargaining code makes it so that)
        # the offering player gets what they proposed and the accepting player receives the remainder.
        if true_state == None:
            return None

        player0_valuations = true_state[self._info["index_start_values"]: self._info["index_start_offers"]]
        player1_valuations = true_state[self._info["index_start_offers"] + self._info["length_each_offer"]: -1]

        assert len(player0_valuations) == len(player1_valuations)

        state_copy = copy.copy(true_state)
        # There is no need to modify the most recent offer. We switch valuations and acting player to capture that this scenario could happen to the other player, which is sufficient.
        if permutation[0] == 0 and permutation[1] == 1:
            # Switch valuations 
            state_copy[self._info["index_start_values"]: self._info["index_start_offers"]] = player0_valuations
            state_copy[self._info["index_start_offers"] + self._info["length_each_offer"]:-1] = player1_valuations
            # Switch who is currently acting
            state_copy[-1] = permutation[state_copy[-1]]
        elif permutation[0] == 1 and permutation[1] == 0:
            # Switch valuations
            state_copy[self._info["index_start_values"]: self._info["index_start_offers"]] = player1_valuations
            state_copy[self._info["index_start_offers"] + self._info["length_each_offer"]:-1] = player0_valuations
            # Switch who is currently acting
            state_copy[-1] = permutation[state_copy[-1]]
        else:
            raise NotImplementedError
        
        return state_copy

    def observations_to_info_state(self, observations):
        # Converts a list of observations for a particular player to an information state (history)
        valuations_pool_and_num_offers = observations[-1][: self._info["index_start_offers"]]

        # NOTE: Is this right? 
        for o in observations[1:]:
            # If there has been at least one offer
            # if any(val > 0 for val in o[1:self._max_turns+1]):
            valuations_pool_and_num_offers = np.hstack([valuations_pool_and_num_offers, o[self._info["index_start_offers"]: ]])
        num_zeros_add = self._info_state_size - len(valuations_pool_and_num_offers)
        assert num_zeros_add >= 0
        info_state = np.hstack([valuations_pool_and_num_offers, np.array([0 for _ in range(num_zeros_add)])])
        return info_state
    
    @property
    def index_start_offers(self):
        return self._info["index_start_offers"]
    
    @property 
    def index_start_values(self):
        return self._info["index_start_values"]

    @property 
    def index_start_pool(self):
        return self._info["index_start_pool"]

    def get_set_info_depending_on_game(self, pyspiel_game):

        self._max_turns = 10 # pyspiel_game.MaxTurns()
        self._pool_max_num_items = 7 # pyspiel_game.kPoolMaxNumItems
        self._num_item_types = 3 # pyspiel_game.kNumItemTypes
        self._total_value_all_items = 10 # pyspiel_game.kTotalValueAllItems

        info = {"index_start_values":2 + self._max_turns + (self._pool_max_num_items + 1) * self._num_item_types}  # Number of elements corresponding to valuations, pool of items and such.
        info["index_start_pool"] = 2 + self._max_turns
        info["index_start_offers"] = info["index_start_values"] + ((self._total_value_all_items + 1) * self._num_item_types) 
        info["length_each_offer"] = (self._pool_max_num_items + 1) * self._num_item_types
        self.set_information(info)
        return 

    