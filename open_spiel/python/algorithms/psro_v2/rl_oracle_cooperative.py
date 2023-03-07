"""
Cooperative oracle for PSRO where we take into account simulation data when adding policies into the strategy set
"""

import numpy as np
from open_spiel.python.algorithms.psro_v2 import rl_oracle
from open_spiel.python.algorithms.psro_v2 import utils
from open_spiel.python.algorithms import imitation
from open_spiel.python.algorithms import imitation_q_learn
from open_spiel.python.algorithms import imitation_deep 
from open_spiel.python.algorithms import imitation_q_learn_deep

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

        self._best_response_class = best_response_class
        self._best_response_kwargs = best_response_kwargs
        

        self._self_play_proportion = self_play_proportion
        self._number_training_steps = number_training_steps
        self._trajectory_returns = [[None] for _ in range(env.num_players)]  # each starts with one policy

        self._consensus_oracle = consensus_kwargs["consensus_oracle"]
        self._imitation_mode = consensus_kwargs["imitation_mode"]
        self._joint = consensus_kwargs["joint"]
        self._num_simulations_fit = consensus_kwargs["num_simulations_fit"]
        self._num_iterations_fit = consensus_kwargs["num_iterations_fit"]
        self.proportion_uniform_trajectories = consensus_kwargs["proportion_uniform_trajectories"]

        self._boltzmann = consensus_kwargs["boltzmann"]
        self._consensus_training_epochs = consensus_kwargs["training_epochs"]
        self._consensus_kwargs = consensus_kwargs

        self._high_return_trajectories = [[] for _ in range(env.num_players)]  # list of lists of trajectories
        self._high_return_actions = [[] for _ in range(env.num_players)]
        self._high_returns = [[] for _ in range(env.num_players)]

        super(RLOracleCooperative, self).__init__(env, best_response_class, best_response_kwargs,
                                                  number_training_steps, self_play_proportion, **kwargs)

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
        steps_per_oracle = [[0
                                for _ in range(len(player_params))]
                               for player_params in training_parameters]
        steps_per_oracle = np.array(steps_per_oracle)

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
            curr_trajectories = rollout_trajectories.get(indexes[0], [])
            curr_returns = rollout_returns.get(indexes[0], [])
            curr_actions = rollout_actions.get(indexes[0], [])

            rollout_trajectories[indexes[0]] = curr_trajectories + [trajectory]
            rollout_returns[indexes[0]] = curr_returns + [returns]
            rollout_actions[indexes[0]] = curr_actions + [actions]

            # Update the number of episodes we have trained per oracle
            steps_per_oracle = rl_oracle.update_steps_per_oracles(steps_per_oracle, indexes, len(actions))

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

        # TODO: List the variety of observations each of the policies were exposed to for reporting 
        
        print("### Calculating the number of distinct observations observed by each policy ### ")

        def count_distinct_observations(buffer):
            set_of_strings = set()
            for trans in buffer:
                set_of_strings.add(np.array2string(np.array(trans.info_state)))
            return len(set_of_strings)
        """
        for i, pol in enumerate(new_policies): 
            buff = pol[-1]._policy.replay_buffer
            all_data = buff.sample(len(buff))
            print("Best response number {} had {} distinct observations".format(i, count_distinct_observations(all_data)))

        for i, pol in enumerate(consensus_policies): 
            buff = pol[-1]._policy.replay_buffer
            all_data = buff.sample(len(buff))
            print("Exploration strategy number {} had {} distinct observations".format(i, count_distinct_observations(all_data)))

        print("")
        """
        return new_policies_total

    def create_consensus_policies(self, training_parameters, rollout_trajectories, rollout_actions, rollout_returns):
        consensus_training_parameters = [[{"policy": None}] for _ in range(len(training_parameters))]
        consensus_policies = self.generate_new_policies(consensus_training_parameters)
        for i in range(len(consensus_policies)):
            # Convert each of the policies to consensus policies. rl_oracle line 230: won't get selected for training
            curr = consensus_policies[i][0]
            new_arguments = {"num_actions": self._best_response_kwargs["num_actions"]}
            if self._consensus_oracle == "trajectory":
                curr._policy = imitation.Imitation(**{"player_id": i, "consensus_kwargs": self._consensus_kwargs}, **new_arguments)
            elif self._consensus_oracle == "q_learn":
                curr._policy = imitation_q_learn.Imitation(**{"player_id": i, "consensus_kwargs": self._consensus_kwargs}, **new_arguments)
            elif self._consensus_oracle == "trajectory_deep":
                new_arguments = {"num_actions": self._best_response_kwargs["num_actions"], 
                                 "state_representation_size": self._best_response_kwargs["state_representation_size"]}
                curr._policy = imitation_deep.Imitation(**{"player_id": i, "consensus_kwargs": self._consensus_kwargs}, **new_arguments)
            elif self._consensus_oracle == "cql_deep":
                new_arguments = {"num_actions": self._best_response_kwargs["num_actions"], 
                                 "state_representation_size": self._best_response_kwargs["state_representation_size"], 
                                 "num_players": self._consensus_kwargs["num_players"]}
                curr._policy = imitation_q_learn_deep.Imitation(**{"player_id": i, "consensus_kwargs": self._consensus_kwargs}, **new_arguments)
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

            # Original code
            max_indices = np.array(social_welfare).argsort()[-self._num_simulations_fit:]

            # Variety trajectories implementation
            max_indices = np.array(social_welfare).argsort()[int(-(1 - self.proportion_uniform_trajectories)*self._num_simulations_fit):]
            low_indices = np.random.randint(low=0, high=len(social_welfare[:int(-(1 - self.proportion_uniform_trajectories)*self._num_simulations_fit)]), size=int(self.proportion_uniform_trajectories*self._num_simulations_fit))
            max_indices = np.concatenate([low_indices, max_indices])

            # print("#### Average social welfare for the last 200 steps: {}".format(np.mean(social_welfare[-200:])), " ####")
            # print("Indices of all trajectories selected: ", max_indices)
            selected_welfare = [social_welfare[k] for k in max_indices]
            print("Selected trajectory stats... min welfare: {}, max welfare: {}, mean welfare: {} ".format(min(selected_welfare), max(selected_welfare), sum(selected_welfare) / len(selected_welfare)))

            curr_return_trajectories = self._high_return_trajectories[player]
            curr_action_trajectories = self._high_return_actions[player]
            curr_returns = self._high_returns[player]

            for k in max_indices:
                curr_return_trajectories.append(rollout_trajectories[(player, policy_index)][k])
                curr_action_trajectories.append(rollout_actions[(player, policy_index)][k])
                curr_returns.append(social_welfare[k])
            # print(curr_return_trajectories, curr_action_trajectories)
            if len(curr_return_trajectories) > (self._num_simulations_fit * self._num_iterations_fit):
                curr_return_trajectories = curr_return_trajectories[-(self._num_simulations_fit * self._num_iterations_fit):]
                curr_action_trajectories = curr_action_trajectories[
                                       -(self._num_simulations_fit * self._num_iterations_fit):]
                curr_returns = curr_returns[-(self._num_simulations_fit * self._num_iterations_fit):]

            self._high_return_trajectories[player] = curr_return_trajectories
            self._high_return_actions[player] = curr_action_trajectories
            self._high_returns[player] = curr_returns

            curr_policy = consensus_policies[player][0]._policy

            print("NUM TRAJECTORIES CONSENSUS FITTING: ", len(curr_return_trajectories))

            for best_trajectory, best_actions, ret in zip(curr_return_trajectories, curr_action_trajectories, curr_returns):
                for i in range(len(best_trajectory) - 1):
                    curr_policy.add_transition(best_trajectory[i], best_actions[i], best_trajectory[i+1], ret)

            print("Training Player {}'s consensus policy".format(player))

            curr_policy.learn()

            if self._consensus_oracle == "optimistic_q":
                new_trajectory_returns.append([np.max(social_welfare)])
            elif self._consensus_oracle == "trajectory":
                new_trajectory_returns.append([np.max(social_welfare)])

            print("\n")

        self.fill_trajectory_returns_with_values(new_trajectory_returns)
        print("Trajectory return values: ", self._trajectory_returns)
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
