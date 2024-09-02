


from open_spiel.python.algorithms.offline_psro.A_pre_training.game_specific_modules.true_state_generator import TrueStateGeneratorWrapper


class LeducPokerTrueStateGenerator(TrueStateGeneratorWrapper):

    def __init__(self, game_name):
        super().__init__(game_name)
        self._true_state_size = 0 # dummy value for now...

    def set_information(self, info):
        self._info = info

    def to_true_state(self, info_states, pyspiel_state):
        # Info_states: the info_states of each respective player 

        # NOTES: 
        # If public_card is -10000, it means it is an invalid card (no public card has been revealed just yet. Pre-flop)
        # We track money instead of pot contributions due to API/Method availability. They both capture the same information.
        curr = []

        # Public Card
        if pyspiel_state.public_card() == -10000:
            curr.append(-1)
        else:
            curr.append(pyspiel_state.public_card())

        # Private Cards
        curr.extend(pyspiel_state.get_private_cards())

        # Pot Contributions / Available Money
        curr.extend(pyspiel_state.money())

        # Current Player
        curr.append(pyspiel_state.current_player())

        # Round Number
        curr.append(pyspiel_state.round())
        return curr

    def get_set_info_depending_on_game(self, pyspiel_game):

        num_suits = 2
        num_players = 2
        total_cards = (num_players + 1) * num_suits
        max_bets_per_round = 3 * num_players - 2
        max_turns = 2 * max_bets_per_round # pyspiel_game.MaxTurns()

        # info = {"index_start_values":2 + max_turns + (pool_max_num_items + 1) * num_item_types}  # Number of elements corresponding to valuations, pool of items and such.
        
        # info["index_start_offers"] = info["index_start_values"] + ((total_value_all_items + 1) * num_item_types) 
        info = {}
        self.set_information(info)
        return 

    