"""
Cooperative oracle for PSRO where we take into account simulation data when adding policies into the strategy set
"""

import numpy as np
from open_spiel.python.algorithms.psro_v2 import rl_oracle
from open_spiel.python.algorithms.psro_v2 import utils
from open_spiel.python.algorithms import imitation
from open_spiel.python.algorithms import imitation_q_learn

class RLOracleCooperative(rl_oracle.RLOracle):

    def __init__(self,
                 env,
                 best_response_class,
                 consensus_oracle,
                 best_response_kwargs,
                 number_training_episodes=1e3,
                 self_play_proportion=0.0,
                 imitation_mode="prob_reward",
                 num_simulations_fit=1,
                 num_iterations_fit=1,
                 joint=True,
                 **kwargs):
        """Init function for the RLOracle.

        Args:
          env: rl_environment instance.
          best_response_class: class of the best response.
          best_response_kwargs: kwargs of the best response.
          number_training_episodes: (Minimal) number of training episodes to run
            each best response through. May be higher for some policies.
          self_play_proportion: Float, between 0 and 1. Defines the probability that
            a non-currently-training player will actually play (one of) its
            currently training strategy (Which will be trained as well).
          **kwargs: kwargs
        """
        # TODO: add the number of simulations to take here
        self._env = env

        self._best_response_class = best_response_class
        self._best_response_kwargs = best_response_kwargs
        self._consensus_oracle = consensus_oracle
        self._imitation_mode = imitation_mode
        self._joint = joint

        self._self_play_proportion = self_play_proportion
        self._number_training_episodes = number_training_episodes
        self._trajectory_returns = [[None] for _ in range(env.num_players)]  # each starts with one policy

        self._num_simulations_fit = num_simulations_fit
        self._num_iterations_fit = num_iterations_fit
        self._high_return_trajectories = [[] for _ in range(env.num_players)]  # list of lists of trajectories
        self._high_return_actions = [[] for _ in range(env.num_players)]

        super(RLOracleCooperative, self).__init__(env, best_response_class, best_response_kwargs,
                                                  number_training_episodes, self_play_proportion, **kwargs)

    def get_trajectory_returns(self):
        return self._trajectory_returns

    def fill_trajectory_returns_with_none(self, all_policies):
        # Match the shape of all_policies by filling in remaining slots with None
        for i, policies in enumerate(all_policies):
            curr_returns = self._trajectory_returns[i]
            num_missing_terms = len(policies) - len(curr_returns)
            curr_returns.extend([None] * num_missing_terms)
        return

    def fill_trajectory_returns_with_values(self, new_trajectory_returns):
        # List of lists denoting the trajectory return for each players' new exploration policies
        for i, returns in enumerate(new_trajectory_returns):
            curr_returns = self._trajectory_returns[i]
            curr_returns.extend(returns)
        return

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
        episodes_per_oracle = [[0
                                for _ in range(len(player_params))]
                               for player_params in training_parameters]
        episodes_per_oracle = np.array(episodes_per_oracle)

        new_policies = self.generate_new_policies(training_parameters)

        all_policies = [old_pol + new_pol for old_pol, new_pol in zip(training_parameters[0][0]["total_policies"], new_policies)]
        self.fill_trajectory_returns_with_none(all_policies)

        rollout_trajectories, rollout_actions, rollout_returns = {}, {}, {}

        """
        print("Clearing observation")
        for agent_index, policy_set in enumerate(all_policies):
            for policy_index, p in enumerate(policy_set):
                if isinstance(p._policy, imitation_q_learn.Imitation):
                    p._policy.running_not_seen_steps = 0
                    p._policy.running_steps = 0
        """

        while not self._has_terminated(episodes_per_oracle):
            """ Note: basically, this while loop cycles through each of the agent's new policies at a time. It will 
            use sample_policies_for_episode to determine which agent's new policy to train and what policies each of the 
            player targets will use. Then, it will do the rollout where it trains the new policy for an episode. Then, 
            update the number of episodes each new policy has been trained. """
            agents, indexes = self.sample_policies_for_episode(
                new_policies, training_parameters, episodes_per_oracle,
                strategy_sampler)

            assert len(indexes) == 1  # we are tailoring this code for one agent training at a time. Make sure true
            assert len(indexes[0]) == 2  # make sure player index and policy index
            assert indexes[0][1] == 0 # make sure this is true because we FIXED it so that PSRO only trains 1 strategy for each player each iteration

            # Store the episode's trajectory and returns and map it to the correct agent + agent's policy we're training
            trajectory, actions, returns = self._rollout(game, agents, **oracle_specific_execution_kwargs)
            curr_trajectories = rollout_trajectories.get(indexes[0], [])
            curr_returns = rollout_returns.get(indexes[0], [])
            curr_actions = rollout_actions.get(indexes[0], [])

            rollout_trajectories[indexes[0]] = curr_trajectories + [trajectory]
            rollout_returns[indexes[0]] = curr_returns + [returns]
            rollout_actions[indexes[0]] = curr_actions + [actions]

            # Update the number of episodes we have trained per oracle
            episodes_per_oracle = rl_oracle.update_episodes_per_oracles(episodes_per_oracle, indexes)

        # Generate one new policy for each agent to train imitation
        # If a dummy is passed in for "policy," a new policy is created and not copied. Assumed that we create one
        # for every player

        """
        print("Printing observation seen proportions: ")
        for agent_index, policy_set in enumerate(all_policies):
            for policy_index, p in enumerate(policy_set):
                if isinstance(p._policy, imitation_q_learn.Imitation):
                    p._policy.running_not_seen_steps = 0
                    p._policy.running_steps = 0
                    print("Proportion of not seen observations for agent_index {} policy index {}".format(agent_index,
                                                                                                          policy_index),
                          float(p._policy.running_not_seen_steps) / p._policy.running_steps)
        """
        print("Creating Consensus Policies: ")
        consensus_policies = self.create_consensus_policies(training_parameters, rollout_trajectories,
                                                         rollout_actions, rollout_returns)

        # After training, concatenate the best response and consensus policies together
        new_policies_total = [new_policies[i] + consensus_policies[i] for i in range(len(new_policies))]

        # Freeze the new policies to keep their weights static. This allows us to
        # later not have to make the distinction between static and training
        # policies in training iterations.
        rl_oracle.freeze_all(new_policies_total)
        return new_policies_total

    def create_consensus_policies(self, training_parameters, rollout_trajectories, rollout_actions, rollout_returns):
        consensus_training_parameters = [[{"policy": None}] for _ in range(len(training_parameters))]
        consensus_policies = self.generate_new_policies(consensus_training_parameters)
        for i in range(len(consensus_policies)):
            # Convert each of the policies to consensus policies. rl_oracle line 230: won't get selected for training
            curr = consensus_policies[i][0]
            new_arguments = {"num_actions": self._best_response_kwargs["num_actions"]}
            if self._consensus_oracle == "trajectory":
                curr._policy = imitation.Imitation(**{"player_id": i, "imitation_mode": self._imitation_mode}, **new_arguments)
            elif self._consensus_oracle == "optimistic_q":
                curr._policy = imitation_q_learn.Imitation(**{"player_id": i, "joint": self._joint}, **new_arguments)
            else:
                print("Pure q_learning is not implemented. Otherwise, inputted an invalid consensus oracle type")
                assert False

        new_trajectory_returns = []
        for player, policy_index in rollout_trajectories.keys():
            # Find the index of the trajectory with highest return for each of the players. Assume 1 policy per player
            returns = rollout_returns[(player, policy_index)]

            # Find the highest self._num_simulations_fit trajectories from each of the past self._num_iterations_fit
            # simulations in terms of social welfare
            social_welfare = [np.sum(individual_returns) for individual_returns in returns]

            max_indices = np.array(social_welfare).argsort()[-self._num_simulations_fit:]
            print("#### Average social welfare for the last 200 steps: {}".format(np.mean(social_welfare[-200:])), " ####")
            print("Indices of all trajectories selected: ", max_indices)
            print("Social welfare of all trajectories selected: ", [social_welfare[k] for k in max_indices])

            curr_return_trajectories = self._high_return_trajectories[player]
            curr_action_trajectories = self._high_return_actions[player]
            for k in max_indices:
                curr_return_trajectories.append(rollout_trajectories[(player, policy_index)][k])
                curr_action_trajectories.append(rollout_actions[(player, policy_index)][k])
            # print(curr_return_trajectories, curr_action_trajectories)
            if len(curr_return_trajectories) > (self._num_simulations_fit * self._num_iterations_fit):
                curr_return_trajectories = curr_return_trajectories[-(self._num_simulations_fit * self._num_iterations_fit):]
                curr_action_trajectories = curr_action_trajectories[
                                       -(self._num_simulations_fit * self._num_iterations_fit):]

            """
            print("Printing BR simulation trajectories from player {} : ".format(player))
            for index, a in enumerate(rollout_actions[(player, policy_index)]):
                if index % 200 == 0:
                    print(index, ": ", a)
            """

            self._high_return_trajectories[player] = curr_return_trajectories
            self._high_return_actions[player] = curr_action_trajectories
            """
            max_index = np.argmax(social_welfare)
            best_trajectory = rollout_trajectories[(player, policy_index)][max_index]
            best_actions = rollout_actions[(player, policy_index)][max_index]
            print("Best action trajectory from player {}'s simulation: ".format(player), best_actions)
            """
            curr_policy = consensus_policies[player][0]._policy
            """if self._consensus_oracle == "trajectory":
                new_trajectory_returns.append([np.max(social_welfare)])
                # Add the trajectory transitions to each of the policies
                all_policies = training_parameters[player][0]["total_policies"][player]
                fit_to_trajectory, fit_to_actions = self.filter_trajectory(best_trajectory, best_actions, all_policies, player)
                best_trajectory, best_actions = fit_to_trajectory, fit_to_actions"""

            print("NUM TRAJECTORIES CONSENSUS FITTING: ", len(curr_return_trajectories))

            # print("ACTION TRAJECTORIES: ")
            for best_trajectory, best_actions in zip(curr_return_trajectories, curr_action_trajectories):
                # print(best_actions, "\n")
                for i in range(len(best_trajectory) - 1):
                    curr_policy.add_transition(best_trajectory[i], best_actions[i], best_trajectory[i+1])

            print("Training Player {}'s consensus policy".format(player))

            curr_policy.learn()

            if self._consensus_oracle == "optimistic_q":
                # Insert the MAX q values from the learned policy instead of the trajectory specific return
                # print("Max social welfare: ", np.max(social_welfare))
                # print("Max Q Value: ", curr_policy.get_max_q_value())
                new_trajectory_returns.append([np.max(social_welfare)])
                # assert len(best_trajectory) == len(best_actions) + 1
                """
                print("Q Values from consensus: ")
                for state_values in curr_policy.q_values:
                    for a, set_to_values in state_values.items():
                        line = [str(set) + ': ' + str(value) + '    ' for set, value in set_to_values.items()]
                        print("Action {}".format(a) + ":  " + ''.join(line))
                """
            elif self._consensus_oracle == "trajectory":
                new_trajectory_returns.append([np.max(social_welfare)])

            print("\n")

        self.fill_trajectory_returns_with_values(new_trajectory_returns)
        print("Trajectory return values: ", self._trajectory_returns)
        return consensus_policies

    """
    def filter_trajectory(self, trajectory, actions, all_policies, player_id):
        obs_set = []
        timestep_set = []  # this is a flatlist same shape as obs_set
        a_set = []
        r_set = []
        for i in range(len(trajectory) - 1):
            o = trajectory[i].observations["info_state"][player_id]
            a = actions[i][player_id]
            if any([np.all(o == past_obs) for past_obs in obs_set]):
                # If in the set
                index = [np.all(o == past_obs) for past_obs in obs_set].index(True)
                past_actions = a_set[index]
                if a not in past_actions:
                    past_actions.append(a)
                    r_set[index].append(-np.inf)
                index_a = past_actions.index(a)
                r_set[index][index_a] = max(r_set[index][index_a], np.sum(trajectory[i+1].rewards))
            else:
                obs_set.append(o)
                timestep_set.append(trajectory[i])
                a_set.append([a])
                r_set.append([np.sum(trajectory[i+1].rewards)])

        # Check which observations have multiple actions. For each, only keep the one with
        # lowest probability other policies execute that action
        fit_to_trajectory = []
        fit_to_actions = []
        for i in range(len(obs_set)):
            actions = a_set[i]
            if len(actions) > 1:
                fit_to_actions.append(actions[np.argmax(r_set[i])])
            else:
                fit_to_actions.append(actions[0])
            fit_to_trajectory.append(timestep_set[i])
        return fit_to_trajectory, fit_to_actions
    """

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
            if time_step.is_simultaneous_move():
                action_list = []
                for agent in agents:
                    output = agent.step(time_step, is_evaluation=is_evaluation)
                    action_list.append(output.action)
                episode_actions.append(action_list)
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

        if not is_evaluation:
            for agent in agents:
                agent.step(time_step)  # This is where agents are trained on the last step

        return episode_trajectory, episode_actions, cumulative_rewards

    def _rollout(self, game, agents, **oracle_specific_execution_kwargs):
        return self.sample_episode(None, agents, is_evaluation=False)
