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
        self.num_players = env.num_players

        self._consensus_oracle = consensus_kwargs["consensus_oracle"]
        self._imitation_mode = consensus_kwargs["imitation_mode"]
        self._rewards_joint = consensus_kwargs["rewards_joint"]
        self._num_simulations_fit = consensus_kwargs["num_simulations_fit"]
        self._num_iterations_fit = consensus_kwargs["num_iterations_fit"]
        self.proportion_uniform_trajectories = consensus_kwargs["proportion_uniform_trajectories"]

        self._boltzmann = consensus_kwargs["boltzmann"]
        self._consensus_training_epochs = consensus_kwargs["training_epochs"]
        self._consensus_kwargs = consensus_kwargs

        self._high_return_trajectories = []  # list of lists of trajectories
        self._high_return_actions = []
        self._high_returns = [[] for _ in range(env.num_players)]

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
        steps_per_oracle = [[0
                                for _ in range(len(player_params))]
                               for player_params in training_parameters]
        steps_per_oracle = np.array(steps_per_oracle)

        new_policies = self.generate_new_policies(training_parameters)

        all_policies = [old_pol + new_pol for old_pol, new_pol in zip(training_parameters[0][0]["total_policies"], new_policies)]
        
        rollout_trajectories, rollout_actions, rollout_returns = {}, {}, {}

        # cutoff_returns = [self._high_returns[i][-1] for i in range(self.num_players)]
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

            # TODO Only insert if the trajectory meets threshold (it's only a candidate if it is greater)
            # if all([cutoff_returns[i] < returns[i] for i in range(self.num_players)]):
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

        print("Creating Consensus Policies: ")

        # TODO: Insert additional rollout_trajectories, actions, and returns from the empirical gamestate updates
        consensus_policies = self.create_consensus_policies(training_parameters, rollout_trajectories,
                                                         rollout_actions, rollout_returns)

        """
        print("Fine tuning consensus policies: ")
        past_returns = []
        for k in range(2500):
            agents = [consensus_policies[i][0] for i in range(len(consensus_policies))]
            if len(agents) == 1: # symmetric
                agents = agents * self.num_players
            _, _, returns = self._rollout(game, agents, **oracle_specific_execution_kwargs)
            past_returns.append(returns)
            if k % (int(25)) == 0 and k != 0:
                print("Average returns: ", np.mean(np.array(past_returns[-25:]), axis=0), np.mean(np.array(past_returns[-25:]), axis=1))
        """

        # After training, concatenate the best response and consensus policies together
        new_policies_total = [new_policies[i] + consensus_policies[i] for i in range(len(new_policies))]

        # Freeze the new policies to keep their weights static. This allows us to
        # later not have to make the distinction between static and training
        # policies in training iterations.
        rl_oracle.freeze_all(new_policies_total)

        def count_distinct_observations(buffer):
            set_of_strings = set()
            for trans in buffer:
                set_of_strings.add(np.array2string(np.array(trans.info_state)))
            return len(set_of_strings)

        return new_policies_total
    
    def fine_tune_policies(self, policies):
        symmetric = len(policies) == 1
        # Create the joint policy module here 
        joint_module = "insert here"
        for k in range(5000):  # this is about 2M steps in the environment
            x = 1
            time_step = self._env.reset()
            cumulative_rewards = 0.0
            steps = 0
            while not time_step.last():
                if time_step.is_simultaneous_move():
                    action_list = joint_module.step(time_step, is_evaluation=is_evaluation)
                    time_step = self._env.step(action_list)
                    cumulative_rewards += np.array(time_step.rewards)
                else:
                    raise NotImplementedError
                steps += 1

            for agent in agents:
                agent.step(time_step)
        return policies  # Should be fine-tuned in-place
    
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
                new_arguments = {"num_actions": self._best_response_kwargs["num_actions"], "state_representation_size": self._consensus_kwargs["state_representation_size"], "num_players": self._consensus_kwargs["num_players"]}
                curr._policy = imitation_deep.Imitation(**{"player_id": i, "consensus_kwargs": self._consensus_kwargs}, **new_arguments)
            elif self._consensus_oracle == "cql_deep":
                new_arguments = {"num_actions": self._best_response_kwargs["num_actions"], "state_representation_size": self._consensus_kwargs["state_representation_size"], "num_players": self._consensus_kwargs["num_players"]}
                curr._policy = imitation_q_learn_deep.Imitation(**{"player_id": i, "consensus_kwargs": self._consensus_kwargs}, **new_arguments)
            else:
                raise NotImplementedError

        is_symmetric = len(training_parameters) == 1 and len(training_parameters) < self.num_players
        
        curr_trajectories = self._high_return_trajectories
        curr_action_trajectories = self._high_return_actions
        was_empty_before = True if len(curr_trajectories) == 0 else False


        for player, policy_index in rollout_trajectories.keys():
            rets = rollout_returns[(player, policy_index)]  # list of tuples (player0 payoff, player1 payoff)
            for r in rets: 
                for i in range(self.num_players):
                    self._high_returns[i].append(r[i])
            curr_trajectories.extend(rollout_trajectories[(player, policy_index)])
            curr_action_trajectories.extend(rollout_actions[(player, policy_index)])

        # Create a list of tuples where (trajectory index, player0 payoff, player1 payoff) of ALL the trajectories (including old ones)
        trajectory_info_list = [tuple([i] + [self._high_returns[player][i] for player in range(self.num_players)]) for i in range(len(curr_trajectories))]

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
        num_simulations_take = self._num_simulations_fit // self.num_players if is_symmetric and (not self._consensus_kwargs["joint_action"]) else self._num_simulations_fit
        sorted_indices = sorted(range(len(rankings)), key=lambda i: rankings[i])
        selected_trajectory_indices = sorted(range(len(rankings)), key=lambda i: rankings[i])[-num_simulations_take:]


        # TODO: This is a test..remove when done 
        incremental_seen_observations_joint = []
        incremental_seen_observations = []
        curr_seen_observations = set()
        seen_joint_observations = set()
        for i in range(len(sorted_indices) - 1, -1, -1):
            trajectory_index = sorted_indices[i]
            trajectory = curr_trajectories[trajectory_index]
            for j in range(len(trajectory)):
                timestep = trajectory[j]
                seen_joint_observations.add(''.join(map(str, timestep.observations["global_state"][0][:])))
                for p in range(self.num_players):
                    curr_obs = timestep.observations["info_state"][p][:]
                    curr_seen_observations.add(''.join(map(str, curr_obs)))
            incremental_seen_observations.append(len(curr_seen_observations))
            incremental_seen_observations_joint.append(len(seen_joint_observations))
        print("Cumulative seen decentralized observations: ", [float(num) / incremental_seen_observations[-1] for num in incremental_seen_observations])
        print("Cumulative seen joint observations: ", [float(num) / incremental_seen_observations_joint[-1] for num in incremental_seen_observations_joint] )

        # For each of the selected trajectory indices 
            # Assign the new self._high_return_trajectories to be from the selected trajectories 
            # Similar for actions and returns 

        self._high_return_trajectories = [curr_trajectories[i] for i in selected_trajectory_indices]
        self._high_return_actions = [curr_action_trajectories[i] for i in selected_trajectory_indices]
        for player in range(self.num_players):
            self._high_returns[player] = [self._high_returns[player][i] for i in selected_trajectory_indices]


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

        # Add each of the transitions to the consensus policies corresponding to each player
        # If the game is symmetric, only player0 will be in rollout_trajectories.keys(). The transitions from each player will be added within the method .add_transition
        for player, _ in rollout_trajectories.keys():
            curr_policy = consensus_policies[player][0]._policy
            for trajectory, action_trajectory in zip(self._high_return_trajectories, self._high_return_actions):
                # TODO: Calculate the rewards to go for each and pass it in as well
                rewards_to_go = [np.zeros(self.num_players) for _ in range(len(trajectory) - 1)]
                curr_rtg = 0.0
                for i in range(len(trajectory) - 1, 0, -1):
                    curr_reward = np.array(trajectory[i].rewards)  # rewards for both players
                    curr_rtg = curr_reward + self._consensus_kwargs["discount"] * curr_rtg
                    rewards_to_go[i-1] = curr_rtg

                for i in range(len(trajectory) - 1):
                    # NOTE: If is_symmetric, then add_transition will add observations/actions from BOTH players already

                    # NOTE: Also insert action_trajectory[i+1]. If it is the last i, then we let action be 0 because it won't be used anyway
                    next_action = action_trajectory[i+1] if (i+1) < len(action_trajectory) else [0 for _ in range(self.num_players)] 
                    curr_policy.add_transition(trajectory[i], action_trajectory[i], trajectory[i+1], next_action, ret=rewards_to_go[i]) 

            print("Training Player {}'s consensus policy".format(player))

            curr_policy.learn()

            print("\n")
    
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
                for i, agent in enumerate(agents):
                    # TODO: Update the player id here
                    # print("Updated agent: ", i)
                    agent.update_player_id(i)
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
            for i, agent in enumerate(agents):
                # Update player_id here
                agent.update_player_id(i)
                agent.step(time_step)  # This is where agents are trained on the last step

        return episode_trajectory, episode_actions, cumulative_rewards
