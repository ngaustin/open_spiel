""" This file generates a new bargaining instance file that enforces symmetric valuations across both players. Symmetry is only valid in 
Bargaining if both the valuations and start turn are interchangeable across players. Ensure that the latter is True in the game settings.
This file generates a list of distinct instances based on several criteria specified below. There is no enforcement that all valuations 
sum to the same value, meaning there could be asymmetric valuation sums ex-interim. """

total_items_in_pool = [5, 6, 7]
item_types = 3
max_valuation = 10
min_valuation = 5
count = 0
num_checked = 0


num_pools = 25
num_valuations = 25
import numpy as np 

all_pools = []

for pool in range(num_pools):
    curr_total_items = np.random.choice(total_items_in_pool)

    max_pools = curr_total_items ** item_types 
    max_pool_list = [i for i in range(max_pools)]
    curr_pool = None
    while not curr_pool:
        curr_candidate_pool = np.random.choice(max_pool_list)
        max_pool_list.remove(curr_candidate_pool)

        curr_candidate_pool = [(curr_candidate_pool // (curr_total_items ** j)) % curr_total_items for j in range(item_types)]

        if sum(curr_candidate_pool) == curr_total_items and all([v > 0 for v in curr_candidate_pool]):
            curr_pool = curr_candidate_pool
    all_pools.append(curr_pool)

all_valuations = []

for val in range(num_valuations):
    max_values = max_valuation ** item_types
    max_value_list = [i for i in range(max_values)]
    curr_valuation = None 
    while not curr_valuation:
        curr_candidate_valuation = np.random.choice(max_value_list)
        max_value_list.remove(curr_candidate_valuation)

        curr_candidate_valuation = [(curr_candidate_valuation // (max_valuation ** m)) % max_valuation for m in range(item_types)]

        if sum(curr_candidate_valuation) > min_valuation and sum(curr_candidate_valuation) < max_valuation and all([v > 0 for v in curr_candidate_valuation]):
            curr_valuation = curr_candidate_valuation 
    all_valuations.append(curr_valuation)

lines = []
for pool in all_pools:
    for player1_valuations in all_valuations:
        for player2_valuations in all_valuations:
            lines.append("{} {} {}".format(','.join([str(v) for v in pool]), ','.join([str(v) for v in player1_valuations]), ','.join([str(v) for v in player2_valuations])))

with open('bargaining_instances_symmetric_{}pool_{}valuations.txt'.format(num_pools, num_valuations), "w") as f:
    f.write('\n'.join(lines))