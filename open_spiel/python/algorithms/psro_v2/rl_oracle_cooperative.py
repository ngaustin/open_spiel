"""
Cooperative oracle for PSRO where we take into account simulation data when adding policies into the strategy set
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from open_spiel.python.algorithms.psro_v2 import rl_oracle
from open_spiel.python.algorithms.psro_v2 import utils
from open_spiel.python.algorithms import imitation
from open_spiel.python.algorithms import imitation_q_learn
from open_spiel.python.algorithms import imitation_deep
from open_spiel.python.algorithms import imitation_q_learn_deep
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import joint_wrapper
from open_spiel.python.algorithms import config
import time

class RLOracleCooperative(rl_oracle.RLOracle):

    def __init__(self,
                 env,
                 best_response_class,
                 best_response_kwargs,
                 consensus_kwargs,
                 number_training_steps=1e4,
                 self_play_proportion=0.0,
                 **kwargs):
        """Init function for the RLOracle.

        Args:
          env: rl_environment instance.
          best_response_class: class of the best response.
          best_response_kwargs: kwargs of the best response.
          number_training_steps: (Minimal) number of environment steps to run
            each best response through. May be higher for some policies.
          self_play_proportion: Float, between 0 and 1. Defines the probability that
            a non-currently-training player will actually play (one of) its
            currently training strategy (Which will be trained as well).
          **kwargs: kwargs
        """
        # TODO: add the number of simulations to take here
        self._env = env
        self._is_turn_based = env.is_turn_based

        self._best_response_class = best_response_class
        self._best_response_kwargs = best_response_kwargs


        self._self_play_proportion = self_play_proportion
        self._number_training_steps = number_training_steps
        self.num_players = env.num_players
        self._consensus_oracle = consensus_kwargs["consensus_oracle"]
        self._imitation_mode = consensus_kwargs["imitation_mode"]
        self._rewards_joint = consensus_kwargs["rewards_joint"]
        self._num_simulations_fit = consensus_kwargs["num_simulations_fit"]
        self.proportion_uniform_trajectories = consensus_kwargs["proportion_uniform_trajectories"]

        self._boltzmann = consensus_kwargs["boltzmann"]
        self._consensus_training_epochs = consensus_kwargs["training_epochs"]
        self._consensus_kwargs = consensus_kwargs
        self._fine_tune_bool = consensus_kwargs["fine_tune"]

        self._most_recent_br_policies = None

        config.ppo_training_data = [[[], [], [], []] for _ in range(env.num_players)]

        self._high_return_trajectories = []  # list of lists of trajectories
        self._high_return_actions = []
        self._high_returns = [[] for _ in range(env.num_players)]
        self._train_br_returns = [[] for _ in range(env.num_players)]

        self._all_seen_observations = set()

        super(RLOracleCooperative, self).__init__(env, best_response_class, best_response_kwargs,
                                                  number_training_steps, self_play_proportion, **kwargs)

    # Override
    def __call__(self,
                 game,
                 training_parameters,
                 strategy_sampler=utils.sample_strategy,
                 **oracle_specific_execution_kwargs):
        """Call method for oracle, returns best responses against a set of policies.

        Args:
          game: The game on which the optimization process takes place.
          training_parameters: A list of list of dictionaries (One list per player),
            each dictionary containing the following fields :
            - policy : the policy from which to start training.
            - total_policies: A list of all policy.Policy strategies used for
              training, including the one for the current player.
            - current_player: Integer representing the current player.
            - probabilities_of_playing_policies: A list of arrays representing, per
              player, the probabilities of playing each policy in total_policies for
              the same player.
          strategy_sampler: Callable that samples strategies from total_policies
            using probabilities_of_playing_policies. It only samples one joint
            set of policies for all players. Implemented to be able to take into
            account joint probabilities of action (For Alpharank)
          **oracle_specific_execution_kwargs: Other set of arguments, for
            compatibility purposes. Can for example represent whether to Rectify
            Training or not.

        Returns:
          A list of list, one for each member of training_parameters, of (epsilon)
          best responses.
        """
        steps_per_oracle = [[0 for _ in range(len(player_params))] for player_params in training_parameters]
        steps_per_oracle = np.array(steps_per_oracle)
        is_symmetric = len(training_parameters) == 1 and len(training_parameters) < self.num_players

        # TODO: Check the number of policies in total_policies of training parameters. If odd, then do BR. If even, do consensus policies
        num_policies_total = len(training_parameters[0][0]["total_policies"][0])
        train_best_response = not (num_policies_total % 2 == 0 and self._consensus_kwargs['consensus_imitation'])

        # NOTE: this is a tester to make sure our comparison with other algorithms is fair
        # if False:
        #     new_policies = self.generate_new_policies(training_parameters)
        # else:

        # TODO: Reset the old and new policy names

        # Generate one new policy for each agent to train imitation
        # If a dummy is passed in for "policy," a new policy is created and not copied. Assumed that we create one
        # for every player
        new_policies = self.create_consensus_policies(training_parameters)

        # NOTE: Changed here for fair comparison
        if not train_best_response:
            print("\nTraining consensus policies on offline data from BR and previous joint simulations: ")
            start = time.time()
            new_policies = self.tune_consensus_policies(new_policies, is_symmetric, self._high_return_trajectories, self._high_return_actions)
            print("Finished offline training after {} seconds.".format(time.time() - start))

            if self._consensus_kwargs["clear_trajectories"]:
                print("Clearing trajectories from previous iteration. ")
                self._high_return_trajectories = []  # list of lists of trajectories
                self._high_return_actions = []
                self._high_returns = [[] for _ in range(self._env.num_players)]
        else:
            self._most_recent_br_policies = [lst[0] for lst in new_policies]

        # Prepares the policies for fine tuning
        if self._fine_tune_bool or train_best_response:
            for i in range(len(new_policies)):
                new_policies[i][0]._policy.set_to_fine_tuning_mode(train_best_response)
                print("Setting to fine tune mode: ")

        if (not self._fine_tune_bool) and not train_best_response:
            rl_oracle.freeze_all(new_policies)
            return new_policies


        all_policies = [old_pol + new_pol for old_pol, new_pol in zip(training_parameters[0][0]["total_policies"], new_policies)]

        rollout_trajectories, rollout_actions, rollout_returns = [], [], []

        cutoff_returns = [self._high_returns[i][-1] if len(self._high_returns[i]) > 0 else -np.inf for i in range(self.num_players)]

        print("\nTraining each of the policies for {} steps. ".format(self._number_training_steps))
        while not self._has_terminated(steps_per_oracle):
            """ Note: basically, this while loop cycles through each of the agent's new policies at a time. It will
            use sample_policies_for_episode to determine which agent's new policy to train and what policies each of the
            player targets will use. Then, it will do the rollout where it trains the new policy for an episode. Then,
            update the number of episodes each new policy has been trained. """
            agents, indexes = self.sample_policies_for_episode(
                    new_policies, training_parameters, steps_per_oracle,
                    strategy_sampler)
            assert len(indexes) == 1  # we are tailoring this code for one agent training at a time. Make sure true
            assert len(indexes[0]) == 2  # make sure player index and policy index
            assert indexes[0][1] == 0 # make sure this is true because we FIXED it so that PSRO only trains 1 strategy for each player each iteration

            # Store the episode's trajectory and returns and map it to the correct agent + agent's policy we're training
            trajectory, actions, returns = self._rollout(game, agents, **oracle_specific_execution_kwargs)
            # To save space, we get rid of the decentralized observations or the joint observations if not needed for consensus training
            for timestep in trajectory:
                if self._consensus_kwargs["joint_action"]:
                    timestep.observations["info_state"] = []
                else:
                    timestep.observations["global_state"] = []

            """
            ###########################################################################################################################
            To save space, the only way that this trajectory would not be considered is if both of its values are less than the cutoff.
            It is not sufficient if only one of its values is lower, unless we are working with a symmetric game. In the case of an
            asymmetric game, since we deal with the "ranking" or "placement" of returns as opposed to the minimum of the actual return
            values, the distribution of returns for any player can be skewed.
            ###########################################################################################################################
            """
            invalid_candidate = (returns[0] < cutoff_returns[0]) or (returns[1] < cutoff_returns[1]) if is_symmetric else (returns[0] < cutoff_returns[0]) and (returns[1] < cutoff_returns[1])
            self._train_br_returns[indexes[0][0]].append(np.array(returns))
            if not invalid_candidate:
                rollout_trajectories.append(trajectory)
                rollout_returns.append(returns)
                rollout_actions.append(actions)

                ############## This is just for space saving on the first iteration ####################
                if len(rollout_returns) > self._num_simulations_fit and is_symmetric:
                    ret_0 = [ret[0] for ret in rollout_returns]
                    ret_1 = [ret[1] for ret in rollout_returns]

                    min_0 = min(ret_0)
                    min_1 = min(ret_1)
                    if min_0 < min_1:
                        take_out_index = ret_0.index(min_0)
                    else:
                        take_out_index = ret_1.index(min_1)

                    rollout_trajectories.pop(take_out_index)
                    rollout_returns.pop(take_out_index)
                    rollout_actions.pop(take_out_index)

            # Update the number of episodes we have trained per oracle

            # TODO: This is an invalid way of calculating number of steps for NON-simultaneous games
            steps_per_oracle = rl_oracle.update_steps_per_oracles(steps_per_oracle, indexes, len(actions))

        if self._consensus_kwargs["consensus_imitation"]:
            self.update_trajectories(training_parameters, rollout_trajectories, rollout_actions, rollout_returns)

        # Freeze the new policies to keep their weights static. This allows us to
        # later not have to make the distinction between static and training
        # policies in training iterations.
        rl_oracle.freeze_all(new_policies)
        return new_policies

    def get_training_returns(self):
        #return [np.array(rets) for rets in self._train_br_returns if rets]
        return [np.array(rets) for rets in self._train_br_returns]

    def create_consensus_policies(self, training_parameters):
        consensus_training_parameters = [[{"policy": None}] for _ in range(len(training_parameters))]
        consensus_policies = self.generate_new_policies(consensus_training_parameters)
        for i in range(len(consensus_policies)):
            # Convert each of the policies to consensus policies. rl_oracle: generate_new_policies. This consensus policy won't be copied
            curr = consensus_policies[i][0]
            new_arguments = {"num_actions": self._best_response_kwargs["num_actions"]}
            if self._consensus_oracle == "trajectory":
                curr._policy = imitation.Imitation(**{"player_id": i, "consensus_kwargs": self._consensus_kwargs}, **new_arguments)
            elif self._consensus_oracle == "q_learn":
                curr._policy = imitation_q_learn.Imitation(**{"player_id": i, "consensus_kwargs": self._consensus_kwargs}, **new_arguments)
            elif self._consensus_oracle == "trajectory_deep":
                new_arguments = {"num_actions": self._best_response_kwargs["num_actions"], "state_representation_size": self._consensus_kwargs["state_representation_size"], "num_players": self._consensus_kwargs["num_players"], "turn_based": self._is_turn_based, "prev_policy":self._most_recent_br_policies[i] if self._most_recent_br_policies else None}
                curr._policy = imitation_deep.Imitation(**{"player_id": i, "consensus_kwargs": self._consensus_kwargs}, **new_arguments)

            elif self._consensus_oracle == "cql_deep":
                new_arguments = {"num_actions": self._best_response_kwargs["num_actions"], "state_representation_size": self._consensus_kwargs["state_representation_size"], "num_players": self._consensus_kwargs["num_players"], "turn_based": self._is_turn_based, "prev_policy":self._most_recent_br_policies[i] if self._most_recent_br_policies else None}
                curr._policy = imitation_q_learn_deep.Imitation(**{"player_id": i, "consensus_kwargs": self._consensus_kwargs}, **new_arguments)
            else:
                raise NotImplementedError
        return consensus_policies

    def update_trajectories(self, training_parameters, rollout_trajectories, rollout_actions, rollout_returns):
        # If the number of training parameters is 1 and we are working in a multi-agent gamge, we are training in the symmetric setting
        is_symmetric = len(training_parameters) == 1 and len(training_parameters) < self.num_players
        # Add the most recently found trajectories
        self._high_return_trajectories.extend(rollout_trajectories)
        self._high_return_actions.extend(rollout_actions)
        for rets in rollout_returns:
            for i in range(self.num_players):
                self._high_returns[i].append(rets[i])

        # Create a list of tuples where (trajectory index, player0 payoff, player1 payoff) of ALL the trajectories (including old ones)
        trajectory_info_list = [tuple([i] + [self._high_returns[player][i] for player in range(self.num_players)]) for i in range(len(self._high_return_trajectories))]

        # Initialize an empty list of trajectory indices
        selected_trajectory_indices = []
        candidate_trajectory_indices = []

        # Sort by all players
        sorted_by_players = [sorted(trajectory_info_list, key=lambda x: (x[i+1], x[0])) for i in range(self.num_players)]

        # Create a list of lists of same length as number of trajectories
        rankings = [[] for _ in range(len(trajectory_info_list))]

        # If symmetric, then we CANNOT use ranking. This is because in the symmetric case, we only train 1 BR, so the
        # Iterate through each of the sorted lists. Insert the "rank" of the trajectory index to the list corresponding to that trajectory
        for player, trajectory_info_by_player in enumerate(sorted_by_players):
            for trajectory_index, info in enumerate(trajectory_info_by_player):
                if is_symmetric:
                    rankings[info[0]].append(info[player+1])
                else:
                    rankings[info[0]].append(trajectory_index)  # we insert the ranking (index i) in the list to the list corresponding to trajectory index (info[0])

        # Take the min rank for each of the trajectories
        rankings = [min(ranking_list) for ranking_list in rankings]

        # Take the top self._num_simulations trajectories in terms of rank
        num_simulations_take = min(self._num_simulations_fit // self.num_players, len(rankings)) if is_symmetric and (not self._consensus_kwargs["joint_action"]) else min(self._num_simulations_fit, len(rankings))
        sorted_indices = sorted(range(len(rankings)), key=lambda i: rankings[i])
        selected_trajectory_indices = sorted(range(len(rankings)), key=lambda i: rankings[i])[-num_simulations_take:]

        # TODO: This is a test to analyze state coverage manually. Remove when done
        incremental_seen_observations_joint = []
        incremental_seen_observations = []
        curr_seen_observations = set()
        # seen_joint_observations = set()
        for i in range(len(sorted_indices) - 1, -1, -1):
            trajectory_index = sorted_indices[i]
            trajectory = self._high_return_trajectories[trajectory_index]
            for j in range(len(trajectory)):
                timestep = trajectory[j]
                # if self._consensus_kwargs["joint_action"] and len(timestep.observations["global_state"]) > 0:
                #     seen_joint_observations.add(''.join(map(str, timestep.observations["global_state"][0][:])))
                if not self._consensus_kwargs["joint_action"]:
                    for p in range(self.num_players):
                        curr_obs = timestep.observations["info_state"][p][:]
                        curr_seen_observations.add(''.join(map(str, curr_obs)))
            incremental_seen_observations.append(len(curr_seen_observations))
            # incremental_seen_observations_joint.append(len(seen_joint_observations))

        self._all_seen_observations = self._all_seen_observations.union(curr_seen_observations)
        if not self._consensus_kwargs["joint_action"]:
            # print("Cumulative seen decentralized observations: ", [float(num) / incremental_seen_observations[-1] for num in incremental_seen_observations])
            print("Covered decentralized observations given the number of trajectories taken (assuming BR had complete coverage): ", float(incremental_seen_observations[num_simulations_take - 1]) / len(self._all_seen_observations))
        # if self._consensus_kwargs["joint_action"] and len(timestep.observations["global_state"]) > 0:
        #     # print("Cumulative seen joint observations: ", [float(num) / incremental_seen_observations_joint[-1] for num in incremental_seen_observations_joint] )
        #     print("Covered joint observations given the number of trajectories taken (assuming BR had complete coverage): ", float(incremental_seen_observations_joint[num_simulations_take - 1]) / incremental_seen_observations_joint[-1])

        # only take the selected trajectories and reassign self._high_return_trajectories
        self._high_return_trajectories = [self._high_return_trajectories[i] for i in selected_trajectory_indices]
        self._high_return_actions = [self._high_return_actions[i] for i in selected_trajectory_indices]
        for player in range(self.num_players):
            self._high_returns[player] = [self._high_returns[player][i] for i in selected_trajectory_indices]

        # Print out metrics for manual inspection
        difference = [abs(self._high_returns[0][i] - self._high_returns[1][i]) for i in range(len(selected_trajectory_indices))]

        if is_symmetric:
            print("Overall selected trajectories have mean payoff {} for both player0 and player1 and mean welfare of {}. Mean absolute difference in return is {}\n\n".format(np.mean(self._high_returns[0] + self._high_returns[1]),
                                                                                                                                           np.mean(np.array(self._high_returns[0]) + np.array(self._high_returns[1])),
                                                                                                                                           np.mean(difference)))
        else:
            print("Overall selected trajectories have mean payoff {} for player0, {} for player1 and mean welfare of {}. Mean absolute difference in return is {}\n\n".format(np.mean(self._high_returns[0]),
                                                                                                                                           np.mean(self._high_returns[1]),
                                                                                                                                           np.mean(np.array(self._high_returns[0]) + np.array(self._high_returns[1])),
                                                                                                                                           np.mean(difference)))

    def tune_consensus_policies(self, consensus_policies, symmetric, high_return_trajectories, high_return_actions, override_symmetric=False):
        # Add each of the transitions to the consensus policies corresponding to each player
        # If the game is symmetric, only player0 will be in rollout_trajectories.keys(). The transitions from each player will be added within the method .add_transition
        player_list = [i for i in range(self.num_players)] if not symmetric else [0]
        for player in player_list:
            curr_policy = consensus_policies[player][0]._policy

            for trajectory, action_trajectory in zip(high_return_trajectories, high_return_actions):
                # Calculate the discounted rewards to go for each and pass it in as well
                curr_policy.add_trajectory(trajectory, action_trajectory)

            print("Training Player {}'s consensus policy".format(player))
            curr_policy.learn()

        return consensus_policies

    def sample_episode(self, unused_time_step, agents, is_evaluation=False):
        time_step = self._env.reset()
        cumulative_rewards = 0.0
        """  For saving trajectories, save a list of timesteps for every episode. A timestep includes:
            observation: list of dicts containing one observations per player, each
              corresponding to `observation_spec()`.
            reward: list of rewards at this timestep, or None if step_type is
              `StepType.FIRST`.
            discount: list of discounts in the range [0, 1], or None if step_type is
              `StepType.FIRST`.
            step_type: A `StepType` value."""
        episode_trajectory = [time_step]
        episode_actions = []
        while not time_step.last():  # This is where the episode is rolled out
            if not self._is_turn_based:
                action_list = []
                for i, agent in enumerate(agents):
                    # We update the player id here because of an issue with symmetric games. This ensures that the
                    # correct player observation is retrieved even when only one policy is used for all players
                    agent.update_player_id(i)
                    output = agent.step(time_step, is_evaluation=is_evaluation)
                    if config.check_new_training_rets():
                        config.ppo_training_data[i] = [config.kl_list, config.entropy_list, config.actor_loss_list, config.value_loss_list]
                        config.kl_list, config.entropy_list, config.actor_loss_list, config.value_loss_list = [], [], [], []
                    action_list.append(output.action)
                episode_actions.append(action_list)
                #Game taking a step to the next timestep with input being the actions that each player takes
                time_step = self._env.step(action_list)  # This is a new timestep that is returned. Replace every time
                episode_trajectory.append(time_step)
                cumulative_rewards += np.array(time_step.rewards)
            else:
                player_id = time_step.observations["current_player"]

                # is_evaluation is a boolean that, when False, lets policies train. The
                # setting of PSRO requires that all policies be static aside from those
                # being trained by the oracle. is_evaluation could be used to prevent
                # policies from training, yet we have opted for adding frozen attributes
                # that prevents policies from training, for all values of is_evaluation.
                # Since all policies returned by the oracle are frozen before being
                # returned, only currently-trained policies can effectively learn.
                agent_output = agents[player_id].step(
                    time_step, is_evaluation=is_evaluation)  # This is where agents are trained each step
                action_list = [agent_output.action]
                time_step = self._env.step(action_list)
                cumulative_rewards += np.array(time_step.rewards)

                episode_trajectory.append(time_step)
                episode_actions.append(action_list)

        if not is_evaluation:
            for i, agent in enumerate(agents):
                # Update player_id here
                agent.update_player_id(i)
                agent.step(time_step)  # This is where agents are trained on the last step
        return episode_trajectory, episode_actions, cumulative_rewards
