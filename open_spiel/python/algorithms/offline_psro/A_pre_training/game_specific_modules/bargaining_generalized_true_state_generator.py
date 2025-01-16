


from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.true_state_generator import TrueStateGeneratorWrapper
import numpy as np
import copy 

class BargainingGeneralizedTrueStateGenerator(TrueStateGeneratorWrapper):

    def __init__(self, game_name, info_state_size):
        super().__init__(game_name)
        self._info_state_size = info_state_size

    def set_information(self, info):
        self._info = info
        print("Info: ", self._info)

    def to_true_state(self, observations, pyspiel_state):
        assert len(observations[0]) > self._info["index_history_start"]
        curr = observations[0]  # we include the most recent offer because that determines the return distribution if the next action is accept
        curr = curr + observations[1][self._info["index_value"][0] : self._info["index_value"][1]]
        curr = curr + [pyspiel_state.current_player()]
        return curr

    def action_index_to_vector_representation(self, index):
        if index == len(self._offer_index_to_division):
            return self._accept_vector
        return self._offer_index_to_division[index]

    def project_to_true_initial_state(self, noisy_initial_state):
        """
        Takes a noisy_initial_state and projects it into the space of "valid" initial states in the game. 
        This validity is not a harsh assumption; we just want representations that make physical sense.
        For example, it makes no sense to start in a state with an agreement already made, having a non-zero number of 
        offers already made, or a player having two different valuations for the same item.  """

        noisy_initial_state_copy = np.copy(noisy_initial_state)

        # Form of the state: 
        # 1 Agreement Reached, 1 Num Offers So Far, {num item types} Pool, {num item types} valuation, {num item types} most recent offer, {num item types} other valuation, 1 current player
        offset = 0
        noisy_initial_state_copy[offset] = 0
        offset += 1

        noisy_initial_state_copy[offset] = 0
        offset += 1 

        # Pool
        noisy_initial_state_copy[offset: offset+self._num_item_types] = np.clip(np.rint(noisy_initial_state_copy[offset: offset+self._num_item_types]), a_min=0, a_max=np.inf)
        offset += self._num_item_types

        # Valuation 1
        noisy_initial_state_copy[offset: offset+self._num_item_types] = np.clip(noisy_initial_state_copy[offset:offset+self._num_item_types], a_min=0, a_max=np.inf)
        offset += self._num_item_types

        # Most Recent Offer
        noisy_initial_state_copy[offset: offset+self._num_item_types] = 0
        offset += self._num_item_types

        # Valuation 2
        noisy_initial_state_copy[offset: offset+self._num_item_types] = np.clip(noisy_initial_state_copy[offset:offset+self._num_item_types], a_min=0, a_max=np.inf)
        offset += self._num_item_types

        # Current Player 
        noisy_initial_state_copy[offset] = np.rint(np.clip(noisy_initial_state_copy[offset], a_min=0, a_max=1))
        starting_player = noisy_initial_state_copy[offset]

        return noisy_initial_state_copy, [int(starting_player)]


    def true_state_symmetric_permute(self, true_state, permutation):
        # Converts a true state to the corresponding permutation. Permutation is a length num_players list 
        # where permutation[i] corresponds to player i becoming player permutation[i] instead.

        # Note: returns depend on the player that made the offer (because the bargaining code makes it so that)
        # the offering player gets what they proposed and the accepting player receives the remainder.
        if true_state == None:
            return None

        # Player0's valuations start as usual
        player0_valuations = true_state[self._info["index_value"][0]: self._info["index_value"][1]]

        # Player1's valuations start after indicating the most recent offer AND does not include the last element which is the current player
        player1_valuations = true_state[self._info["index_value"][1] + self._info["length_each_offer"]: -1]

        assert len(player0_valuations) == len(player1_valuations)

        state_copy = copy.copy(true_state)
        # There is no need to modify the most recent offer. We switch valuations and acting player to capture that this scenario could happen to the other player, which is sufficient.
        if permutation[0] == 0 and permutation[1] == 1:
            # Switch valuations 
            state_copy[self._info["index_value"][0]: self._info["index_value"][1]] = player0_valuations
            state_copy[self._info["index_value"][1] + self._info["length_each_offer"]:-1] = player1_valuations
            # Switch who is currently acting
            state_copy[-1] = permutation[state_copy[-1]]
        elif permutation[0] == 1 and permutation[1] == 0:
            # Switch valuations
            state_copy[self._info["index_value"][0] : self._info["index_value"][1]] = player1_valuations
            state_copy[self._info["index_value"][1] + self._info["length_each_offer"]:-1] = player0_valuations
            # Switch who is currently acting
            state_copy[-1] = permutation[state_copy[-1]]
        else:
            raise NotImplementedError
        
        return state_copy

    def observations_to_info_state(self, observations):
        # Converts a list of observations for a particular player to an information state (history)
        valuations_pool_and_num_offers = observations[-1][: self._info["index_history_start"]]

        # Make sure that the first observation has no information on history. Commented out because initial observations could be outputted by neural net, meaning not equal to 0
        # print("Check sum: ", np.sum(observations[0][self._info["index_history_start"]: ]))
        # assert np.sum(observations[0][self._info["index_history_start"]: ]) == 0

        for o in observations[1:]:
            valuations_pool_and_num_offers = np.hstack([valuations_pool_and_num_offers, o[self._info["index_history_start"]: ]])
        num_zeros_add = self._info_state_size - len(valuations_pool_and_num_offers)
        assert num_zeros_add >= 0
        info_state = np.hstack([valuations_pool_and_num_offers, np.array([0 for _ in range(num_zeros_add)])])
        return info_state

    def get_set_info_depending_on_game(self, pyspiel_game):

        self._max_turns = pyspiel_game.max_num_offers
        self._num_item_types = pyspiel_game.item_types

        info = {
                "index_pool": tuple([2,2+self._num_item_types]),
                "index_value": tuple([2+self._num_item_types, 2+2*self._num_item_types]),
                "index_history_start": 2+2*self._num_item_types, 
                "length_each_offer": self._num_item_types
        }
        self.set_information(info)

        self._offer_index_to_division = pyspiel_game.offer_index_to_division
        self._accept_vector = pyspiel_game.get_accept_vector()
        return 

    