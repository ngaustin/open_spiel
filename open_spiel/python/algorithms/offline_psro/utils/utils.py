import collections
import numpy as np
import copy 


####### Rollout Generation ###########

Transition = collections.namedtuple(
    "Transition",
    "info_states actions legal_actions_masks rewards done relevant_players")

# Relevant_players refers to which players should use this Transition to train their policy (because it indicates some RL Timestep from their point of view)

TransitionForTraining = collections.namedtuple(
    "TransitionForTraining",
    "info_state global_state action legal_actions_mask reward next_info_state next_global_state done next_legal_actions_mask between_info_states between_actions between_players"
)
# Between_[info_states, actions, players] refers to the info_states and actions of other players between the two points of time in which our player of interest acted
# This is used to help determine the probability of such a transition by accounting for other players' changing policies (and allowing) other players to take an 
# arbitrary number of actions between this player's two actions. We include other players at the first endpoint to account for the case where players act simultaneously. 

StateAction = collections.namedtuple(
    "StateAction",
    "info_states actions legal_actions_masks global_states"
)

# Not to be edited directly 
DefaultPolicyInfoDict = {"importance_sampled_return": None, "player_id": None, "policy_id": None, "target_id": None, 
                        "corrected_return": None, "q_calculated_return": None}

# Default Policy Evaluation Information
def get_new_default_policy_evaluation_information(num_players, num_evaluation_policies):
    new_dictionary = {"policy_{}".format(pol): [None for player in range(num_players)] for pol in range(num_policies)}
    return new_dictionary

def get_new_default_policy_info_dict():
    return DefaultPolicyInfoDict.copy()

def get_agent_action(time_step, player, policies):
    observation = time_step.observations["info_state"][player]
    legal_actions = time_step.observations["legal_actions"][player]
    return observation, legal_actions, policies[player].step(observation, legal_actions)

def generate_single_rollout(num_players, env, policies, is_turn_based):
    """
    Generate and return a single trajectory. Each trajectory is a list of Transitions, where each transition describes a step from the perspective of one agent.
    In other words, the info_state and next_info_state are both points in time where at least one of the relevant_players are acting. 
    Global_states encapsulate information from all players and describes "all information present" at each of the respective timesteps in the transition. 
    Global_states can be indexed by player simply to be able to track which global_state is attached to which timestep. However, the information is not necessarily solely
    describing any one player. 

    return: a list of Transitions
    """
    
    rollout_for_policy_evaluation = []  # Contains a rollout where each element is a Transition: observations, actions, rewards, next_states, done, legal_actions_masks, players_acting
    rollouts_for_policy_training = [[] for _ in range(num_players)]  # Contains similar rollouts for each player to train on.

    prev_state_action = StateAction(info_states=[None for _ in range(num_players)], 
                                    actions=[None for _ in range(num_players)], 
                                    legal_actions_masks=[None for _ in range(num_players)], 
                                    global_states=[None for _ in range(num_players)])


    time_step = env.reset()

    all_players_acting = []
    all_info_states = []
    all_actions = []

    # Rollout generation and tracking
    while not time_step.last():
        players_acting_this_step = [time_step.observations["current_player"]] if is_turn_based else [i for i in range(num_players)]

        obs, legal_actions, actions, global_states = [None for _ in range(num_players)], [None for _ in range(num_players)], [None for _ in range(num_players)], [None for _ in range(num_players)]
        
        # all_[] variables are each lists of lists (corresponding to the information at each timestep)
        all_players_acting.append(players_acting_this_step)
        all_info_states.append([])
        all_actions.append([])

        # Track observations, actions, and legal actions for all players whose turn it is (could be simultaneous/all players)
        for player in players_acting_this_step: 
            curr_obs, curr_legal_actions, curr_action = get_agent_action(time_step, player, policies)
            obs[player], legal_actions[player], actions[player] = curr_obs, curr_legal_actions, curr_action
            global_states[player] = [item for info_state in time_step.observations["info_state"] for item in info_state]

            all_info_states[-1].append(curr_obs)
            all_actions[-1].append(curr_action)
        
        # Adding Transition objects if the criteria is met 
        acting_players_with_previous_action = [p for p in players_acting_this_step if prev_state_action.actions[p] != None and prev_state_action.info_states[p] != None]

        if len(acting_players_with_previous_action) > 0:
            # Add the transition for policy evaluation
            rollout_for_policy_evaluation.append(Transition(
                info_states=[s if p in acting_players_with_previous_action else None for p, s in enumerate(prev_state_action.info_states)],
                actions=[a if p in acting_players_with_previous_action else None for p, a in enumerate(prev_state_action.actions)],
                legal_actions_masks=[mask if p in acting_players_with_previous_action else None for p, mask in enumerate(prev_state_action.legal_actions_masks)],
                rewards=time_step.rewards,
                done=0,
                relevant_players=acting_players_with_previous_action,
            ))
 
            # Add transitions for policy training (which are separated based on training player)
            for p in acting_players_with_previous_action:
                # Find the last TWO times in which this player acted 
                last_two_times_player_acted = [index for index, acting_player_list in enumerate(all_players_acting) if p in acting_player_list][-2:]

                between_info_states, between_actions, between_acting_players = [], [], []

                # Concatenate all of the info_states not observed by p, corresponding actions, and executing players leading up to that point (including start)
                for time in range(last_two_times_player_acted[0], last_two_times_player_acted[1]):
                    curr_players_acting = all_players_acting[time]
                    for j, player_inner_loop in enumerate(curr_players_acting):
                        if player_inner_loop != p:
                            between_info_states.append(all_info_states[time][j])
                            between_actions.append(all_actions[time][j])
                            between_acting_players.append(player_inner_loop)

                # Insert TransitionForTraining to respective player p's buffer
                rollouts_for_policy_training[p].append(TransitionForTraining(
                    info_state=prev_state_action.info_states[p],
                    global_state=global_states[p],
                    action=prev_state_action.actions[p],
                    legal_actions_mask=prev_state_action.legal_actions_masks[p],
                    reward=time_step.rewards[p],
                    next_info_state=obs[p],
                    next_global_state=global_states[p],
                    done=0,
                    next_legal_actions_mask=[1 if i in legal_actions[p] else 0 for i in range(policies[p].num_actions)],
                    between_info_states=between_info_states,
                    between_actions=between_actions,
                    between_players=between_acting_players
                ))

        # Update the previous state_action pairs for each of the players_acting_this_step
        for player in players_acting_this_step:
            prev_state_action.info_states[player] = obs[player]
            prev_state_action.actions[player] = actions[player]
            prev_state_action.legal_actions_masks[player] = [1 if i in legal_actions[player] else 0 for i in range(policies[player].num_actions)]
            prev_state_action.global_states[player] = global_states[player]

        # step using the action list (should be length 1 or length num players)
        action_list = [a for a in actions if a != None]
        time_step = env.step(action_list)

    # Account for the last step. This should be very similar to what we had before
    # We create one Transition datapoint for each player. That way, we can distinguish between relevant players
    rollout_for_policy_evaluation.append(Transition(
        info_states=prev_state_action.info_states,
        actions=prev_state_action.actions,
        legal_actions_masks=prev_state_action.legal_actions_masks,
        rewards=time_step.rewards, 
        done=1,
        relevant_players=[i for i in range(num_players)],
    ))

    # Add the last transitions for policy training. Note that this is quite different from the additions during the episode!!!
    for p in range(num_players):
        # Find the LAST TIME in which this player acted 
        last_time_player_acted = [index for index, acting_player_list in enumerate(all_players_acting) if p in acting_player_list][-1]

        between_info_states, between_actions, between_acting_players = [], [], []

        # Concatenate all of the info_states not observed by p, corresponding actions, and executing players leading up to that point (including start)
        assert last_time_player_acted < len(all_players_acting)
        for i in range(last_time_player_acted, len(all_players_acting)):
            curr_players_acting = all_players_acting[i]
            for j, player_inner_loop in enumerate(curr_players_acting):
                if player_inner_loop != p:
                    between_info_states.append(all_info_states[i][j])
                    between_actions.append(all_actions[i][j])
                    between_acting_players.append(player_inner_loop)

        # Insert TransitionForTraining to respective player p's buffer
        rollouts_for_policy_training[p].append(TransitionForTraining(
            info_state=prev_state_action.info_states[p],
            global_state=global_states[p],
            action=prev_state_action.actions[p],
            legal_actions_mask=prev_state_action.legal_actions_masks[p],
            reward=time_step.rewards[p],
            next_info_state=time_step.observations["info_state"][p],
            next_global_state=[item for info_state in time_step.observations["info_state"] for item in info_state],
            done=1,
            next_legal_actions_mask=[0 for i in range(policies[p].num_actions)],
            between_info_states=between_info_states,
            between_actions=between_actions,
            between_players=between_acting_players
        ))

    return rollout_for_policy_evaluation, rollouts_for_policy_training

###### Rollout Generation End ###########

