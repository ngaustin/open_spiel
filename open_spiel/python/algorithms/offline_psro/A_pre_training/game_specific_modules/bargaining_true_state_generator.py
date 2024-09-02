


from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.true_state_generator import TrueStateGeneratorWrapper
import numpy as np
import copy 

class BargainingTrueStateGenerator(TrueStateGeneratorWrapper):

    def __init__(self, game_name, info_state_size):
        super().__init__(game_name)
        self._info_state_size = info_state_size

    def set_information(self, info):
        self._info = info
        print("INfo: ", self._info)

    def to_true_state(self, observations, pyspiel_state):
        assert len(observations[0]) > self._info["index_start_offers"]
        curr = observations[0]  # we include the most recent offer because that determines the return distribution if the next action is accept
        curr = curr + observations[1][self._info["index_start_values"] : self._info["index_start_offers"]]
        curr = curr + [pyspiel_state.current_player()]
        return curr

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
        pool_max_num_items = 7 # pyspiel_game.kPoolMaxNumItems
        self.num_item_types = 3 # pyspiel_game.kNumItemTypes
        total_value_all_items = 10 # pyspiel_game.kTotalValueAllItems

        info = {"index_start_values":2 + self._max_turns + (pool_max_num_items + 1) * self.num_item_types}  # Number of elements corresponding to valuations, pool of items and such.
        info["index_start_pool"] = 2 + self._max_turns
        info["index_start_offers"] = info["index_start_values"] + ((total_value_all_items + 1) * self.num_item_types) 
        info["length_each_offer"] = (pool_max_num_items + 1) * self.num_item_types
        self.set_information(info)
        return 

    