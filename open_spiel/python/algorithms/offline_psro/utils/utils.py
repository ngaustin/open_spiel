import collections
import numpy as np
import copy 
import os 
import pickle
import tensorflow.compat.v1 as tf
from absl import logging


class Normalizer:
    def __init__(self, mean=0, standard_deviation=1):
        self.mean = mean 
        self.standard_deviation = standard_deviation 
    
    def modify(self, mean, standard_deviation):
        self.mean = mean
        self.standard_deviation = standard_deviation
    
    def __str__(self):
        return "Mean: {}  StdDev: {}".format(self.mean, self.standard_deviation)



############## Context Managers ################
def get_graphs_and_context_managers(trial_data_path, player_trial_strategy_path, tf_model_manager):
        # for all of the players, load up the relevant graphs
        all_frozen_graphs = []
        all_variable_names = []
        for path in player_trial_strategy_path:
            # List files in player_trial_strategy_path
            full_path = trial_data_path + path
            file_names = [f for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f))]

            # Then, filter for the ones that end in .pb and contain the word "policy"
            frozen_policy_files = [f for f in file_names if ("policy" in f) and (".pb" in f)]

            # Sort by index of the policy
            # TODO: Fix this query for index lambda so that we account for multi-digit numbers
            query_for_index = lambda file_name: (int([token.split('.')[0] for token in file_name.split('_') if token[0].isdigit()][0]))
            sorted_policy_files = sorted(frozen_policy_files, key=query_for_index)

            # Save all graph objects by calling tf_model_manager.load_frozen_graph() and saving to a list
            all_frozen_graphs.append([tf_model_manager.load_frozen_graph(full_path, policy_file) for policy_file in sorted_policy_files])

            variable_name_files = [f for f in file_names if ("variable" in f) and ".pkl" in f]
            sorted_variable_name_files = sorted(variable_name_files, key=query_for_index)

            curr_variable_names = []
            for f in sorted_variable_name_files:
                total_path = trial_data_path + path + f
                with open(total_path, 'rb') as f:
                    variable_names = pickle.load(f)
                    curr_variable_names.append(variable_names)
            all_variable_names.append(curr_variable_names)

            #print(sorted_policy_files)
            #print(sorted_variable_name_files)
        # Load all context managers
        context_managers = [[tf.Session(graph=g) for g in player_set] for player_set in all_frozen_graphs]

        return all_frozen_graphs, context_managers, all_variable_names

####### Rollout Generation ###########

Transition = collections.namedtuple(
    "Transition",
    "info_states actions legal_actions_masks next_info_states rewards done relevant_players global_state next_global_state")

# Relevant_players refers to which players should use this Transition to train their policy (because it indicates some RL Timestep from their point of view)

StateAction = collections.namedtuple(
    "StateAction",
    "info_states actions legal_actions_masks global_state relevant_players"
)

Step = collections.namedtuple(
    "Step",
    "info_state reward is_terminal legal_actions_mask acting_players global_state")

# Not to be edited directly 
DefaultPolicyInfoDict = {"importance_sampled_return": None, "player_id": None, "policy_id": None, "target_id": None, 
                        "corrected_return": None, "q_calculated_return": None}

# Default Policy Evaluation Information
def get_new_default_policy_evaluation_information(num_players, num_evaluation_policies):
    new_dictionary = {"policy_{}".format(pol): [None for player in range(num_players)] for pol in range(num_policies)}
    return new_dictionary

def get_new_default_policy_info_dict():
    return DefaultPolicyInfoDict.copy()

def get_agent_action(time_step, player, policies, env, sessions, info_state=None, manually_pass_info_state=False):
    if manually_pass_info_state:
        state_representation = info_state 
    else:
        state_representation =  time_step.observations["info_state"][player]
    legal_actions = time_step.observations["legal_actions"][player]
    step_dummy = Step(info_state=state_representation, legal_actions_mask=[1 if i in legal_actions else 0 for i in range(env.action_spec()["num_actions"])], 
                    reward=None, is_terminal=None, acting_players=[], global_state=None)
    return policies[player].step(step_dummy, player, session=sessions[player], is_evaluation=True)[0]

def get_terminal_state(N, state_size):
    return np.ones((N, state_size)) * -1

def generate_single_rollout(num_players, env, policies, sessions, is_turn_based, true_state_extractor, start_action=None, manually_create_info_states_from_observations=False):
    """
    Generate and return a single trajectory. Each trajectory is a list of Transitions, where each transition describes a step from the perspective of one agent.
    In other words, the info_state and next_info_state are both points in time where at least one of the relevant_players are acting. 
    Global_states encapsulate information from all players and describes "all information present" at each of the respective timesteps in the transition. 
    Global_states can be indexed by player simply to be able to track which global_state is attached to which timestep. However, the information is not necessarily solely
    describing any one player. 

    return: a list of Transitions
    """
    if manually_create_info_states_from_observations:
        assert env._use_observation 
    
    rollout = []
    prev_state_action = StateAction(info_states=[None for _ in range(num_players)], 
                                    actions=[None for _ in range(num_players)], 
                                    legal_actions_masks=[None for _ in range(num_players)], 
                                    relevant_players=[],
                                    global_state=None)

    time_step = env.reset(manual_sample=start_action)
    
    # Rollout generation and tracking
    obs, legal_actions, actions = [None for _ in range(num_players)], [None for _ in range(num_players)], [None for _ in range(num_players)]
    ################ For testing/debugging ######################
    if env.is_turn_based:
        # For turn based games, observations from ALL players carry the information needed to reconstruct info_states for each player
        starting_player = time_step.observations["current_player"]
        all_observations = [[time_step.observations["info_state"][starting_player]] for p in range(num_players)] 
        info_state = true_state_extractor.observations_to_info_state(all_observations[starting_player])
    else:
        raise NotImplementedError
    #############################################################
    while not time_step.last():
        players_acting_this_step = [time_step.observations["current_player"]] if is_turn_based else [i for i in range(num_players)]

        # Track observations, actions, and legal actions for all players whose turn it is (could be simultaneous/all players)
        for player in range(num_players): 
            if player in players_acting_this_step:
                curr_legal_actions = time_step.observations["legal_actions"][player]

                if manually_create_info_states_from_observations:
                    info_state = true_state_extractor.observations_to_info_state(all_observations[player])
                    curr_action = get_agent_action(time_step, player, policies, env, sessions, info_state, manually_create_info_states_from_observations)
                else:
                    curr_action = get_agent_action(time_step, player, policies, env, sessions)

                if curr_action not in curr_legal_actions:
                    print("Action {} is not one of the legal actions: {}".format(curr_action, curr_legal_actions))
                    raise Exception
            else:
                curr_action = None 
                curr_legal_actions = []
            
            curr_obs = time_step.observations["info_state"][player]
            obs[player], legal_actions[player], actions[player] = curr_obs, curr_legal_actions, curr_action
        
        global_state = true_state_extractor.to_true_state(time_step.observations["info_state"], env._state)

        if any([a != None for a in prev_state_action.actions]):  # if any actions are not None, then we have a previous timestep
            # Add the timeStep
            rollout.append(Transition(
                info_states=prev_state_action.info_states,
                actions=prev_state_action.actions,
                legal_actions_masks=prev_state_action.legal_actions_masks,
                rewards=time_step.rewards,
                next_info_states=copy.copy(obs),
                done=0,
                relevant_players=prev_state_action.relevant_players,
                global_state=prev_state_action.global_state,
                next_global_state=global_state,
            ))
 
        # Update the previous state_action pairs for each of the players_acting_this_step
        prev_state_action=StateAction(
            info_states=copy.copy(obs),
            actions=copy.copy(actions),
            legal_actions_masks=[[1 if i in legal_actions[player] else 0 for i in range(policies[player]._num_actions)] for player in range(num_players)],
            relevant_players=players_acting_this_step,
            global_state=global_state
        )

        # step using the action list (should be length 1 or length num players)
        action_list = [a for player, a in enumerate(actions) if player in players_acting_this_step]
        time_step = env.step(action_list)
        
        if not time_step.last():
            for p in range(num_players):
                all_observations[p].append(time_step.observations["info_state"][time_step.observations["current_player"]])
                
    # Account for the last step. This should be very similar to what we had before
    rollout.append(Transition(
        info_states=prev_state_action.info_states,
        actions=prev_state_action.actions,
        legal_actions_masks=prev_state_action.legal_actions_masks,
        rewards=time_step.rewards, 
        next_info_states=[None for _ in range(num_players)],
        done=1,
        relevant_players=prev_state_action.relevant_players,
        global_state=prev_state_action.global_state,
        next_global_state=None
    ))

    return rollout # , rollouts_for_policy_training, first_steps_for_policy_training

###### Rollout Generation End ###########


##### Hashing for Discrete Policies #####

def compute_hash_string(state):
    return '_'.join([str(item) for item in state])

def inverse_hash_string(string):
    return [float(s) for s in string.split('_')]

#### Hashing for Discrete Policies End ##

