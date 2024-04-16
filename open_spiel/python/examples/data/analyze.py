

import numpy as np
from absl import app
from absl import flags
import os

import matplotlib.pyplot as plt

#RRD Runner
from open_spiel.python.algorithms import projected_replicator_dynamics
import tensorflow.compat.v1 as tf
import sys
import time

FLAGS = flags.FLAGS

flags.DEFINE_string("game_data_path",  "", "Directory for retrieving game data. Assuming iteration structure.")
flags.DEFINE_string("save_graph_path",  "", "Directory for saving graphs. Directory will be created if doesn't exist.")
flags.DEFINE_string("game_name",  "Harvest", "Name of game.")
flags.DEFINE_integer("n_players", 2, "Number of players")
flags.DEFINE_boolean("graph_mode", True, "Will/won't graph.")
flags.DEFINE_boolean("is_symmetric", True, "Symmetric game?")
flags.DEFINE_boolean("with_std", True, "Use std error bars?")
flags.DEFINE_boolean("expsro", False, "ExPSRO Run?")
flags.DEFINE_boolean("truncate", False, "Truncate the runs for closer analysis?")

# NOTE: File is designed to analyze one game at a time.

# For the simple one state game:
#    - Policies and meta probabilities for each step
#    - Overall Payoff matrix at the very end

# For the three state game:
#    - Overall payoff matrix
#    -

def _partial_multi_dot(player_payoff_tensor, strategies, index_avoided):
  """Computes a generalized dot product avoiding one dimension.

  This is used to directly get the expected return of a given action, given
  other players' strategies, for the player indexed by index_avoided.
  Note that the numpy.dot function is used to compute this product, as it ended
  up being (Slightly) faster in performance tests than np.tensordot. Using the
  reduce function proved slower for both np.dot and np.tensordot.

  Args:
    player_payoff_tensor: payoff tensor for player[index_avoided], of dimension
      (dim(vector[0]), dim(vector[1]), ..., dim(vector[-1])).
    strategies: Meta strategy probabilities for each player.
    index_avoided: Player for which we do not compute the dot product.

  Returns:
    Vector of expected returns for each action of player [the player indexed by
      index_avoided].
  """
  new_axis_order = [index_avoided] + [
      i for i in range(len(strategies)) if (i != index_avoided)
  ]
  accumulator = np.transpose(player_payoff_tensor, new_axis_order)
  for i in range(len(strategies) - 1, -1, -1):
    if i != index_avoided:
      accumulator = np.dot(accumulator, strategies[i])
  return accumulator

def main(argv):
  """ Load and analyze a list of numpy arrays representing:
        [num_players, S] shape matrix of meta_probabilities (where S is the number of policies so far)
        [num_players, num_states, num_actions, S] shape matrix representing policies
        [num_players, S, S] shape matrix representing utilities (U)

     All three of these are contained in each of the interation .npy files for each iteration
  """
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  
  # Parent directory for subdirectories containing trial game data
  data_directory_path = os.getcwd() + FLAGS.game_data_path
  name_of_method = FLAGS.game_name
  save_graph_path = FLAGS.save_graph_path
  graph_mode = FLAGS.graph_mode
  is_symmetric = FLAGS.is_symmetric
  num_players = FLAGS.n_players if not is_symmetric else 1

  def graph_kl(kl_values, folder_path):
    #Graph the kl_divergence values over iterations
    kl_fig, ax = plt.subplots()
    for num_player in range(num_players):  
      ax.scatter(x=[ind for ind in range(len(kl_values[num_player]))],
        y=[np.mean(np.array(kl_value_iter)) for kl_value_iter in kl_values[num_player]],
        label="{}: Mean kl_values".format(name_of_method))
      ax.set_title("Mean KL Values over Iterations")
      ax.set_xlabel("Iteration")
      ax.set_ylabel("Mean KL Values")
      _, end = ax.get_xlim()
      ax.xaxis.set_ticks(np.arange(0, end, 1))
      kl_fig.savefig(folder_path + "mean_kl_player_{}.jpg".format(num_player))
    plt.close()

  def graph_training_returns(training_returns, num_iteration, folder_path, average_grouping = 1000):
    """
      Graph training returns for BR training
        training_returns: 2D array with n arrays containing training returns for players
        num_iteration: # of current iteration
        average_grouping: Size of episode grouping (for cleaner graphing purposes)
        axis_normalization: Normalizes y-axis scale across the iterations for easier trend comparisons
    """
    if not os.path.exists(folder_path + "training_returns/"):
      os.makedirs(folder_path + "training_returns/")
    grouped_rets = []
    curr_player = 0
    #Entire training returns dataset
    for i in range(len(training_returns)):
      if len(training_returns[i]):
        grouped_rets.append([])
        #Each BR Policy
        for j in range(0, len(training_returns[i]) - average_grouping + 1):
          #Breakdown of BR policy returns for each player
          grouped_rets[i].append(np.mean(training_returns[i][j: j + average_grouping, curr_player]))
      curr_player += 1
    curr_player = 0
    for i in range(len(grouped_rets)):
        training_returns_fig, ax = plt.subplots()
        ax.plot(np.arange(0, len(grouped_rets[i])), grouped_rets[i])
        ax.set_title("Iteration {}: BR Policy {} Training Returns Player {}".format(num_iteration, i + 1, curr_player + 1))
        ax.set_xlabel("Iteration (Grouped by {} episodes)".format(average_grouping))
        ax.set_ylabel("Expected Returns")
        training_returns_fig.savefig(folder_path + "training_returns/" + "policy_{}_iteration_{}_player_{}.jpg".format(i + 1, num_iteration, curr_player + 1))
    plt.close()
  
  def graph_regret_training_returns(training_returns, num_iteration, folder_path, average_grouping = 200):
    """
      Graph regret training returns for BR training
        training_returns: 2D array with n arrays containing training returns for players
        num_iteration: # of current iteration
        average_grouping: Size of episode grouping (for cleaner graphing purposes)
        axis_normalization: Normalizes y-axis scale across the iterations for easier trend comparisons
    """
    if not os.path.exists(folder_path + "training_regret_returns/"):
      os.makedirs(folder_path + "training_regret_returns/")
    grouped_rets = []
    curr_player = 0
    #Entire training returns dataset
    for i in range(len(training_returns)):
      if len(training_returns[i]):
        grouped_rets.append([])
        #Each BR Policy
        for j in range(0, len(training_returns[i]) - average_grouping + 1):
          #Breakdown of BR policy returns for each player
          grouped_rets[i].append(np.mean(training_returns[i][j: j + average_grouping, curr_player]))
      curr_player += 1
    curr_player = 0
    for i in range(len(grouped_rets)):
        training_returns_fig, ax = plt.subplots()
        ax.plot(np.arange(0, len(grouped_rets[i])), grouped_rets[i])
        ax.set_title("Iteration {}: BR Policy {} Training Regret Returns Player {}".format(num_iteration, i + 1, curr_player + 1))
        ax.set_xlabel("Iteration (Grouped by {} episodes)".format(average_grouping))
        ax.set_ylabel("Expected Returns")
        training_returns_fig.savefig(folder_path + "training_regret_returns/" + "policy_{}_iteration_{}_player_{}.jpg".format(i + 1, num_iteration, curr_player + 1))
    plt.close()

  def graph_max_welfare(data, folder_path):
    #Graph the max social welfare over iterations
    welfare_fig, ax = plt.subplots()
    ax.scatter(x=[ind for ind in range(len(data))],
      y=data,
      label="{}: max welfare".format(name_of_method))
    ax.set_title("Max Welfare Over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Welfare")
    _, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, 3))
    welfare_fig.savefig(folder_path + "max_welfare.jpg")
    plt.close()

  def graph_expected_welfare(data, names, folder_path):
    # Graph the expected welfare over iterations
    welfare_fig, ax = plt.subplots()
    line_colors, display_order = get_color_gradient_for_ex2psro(names) if FLAGS.expsro else None
    x = [ind for ind in range(len(data[0][0]))]
    if FLAGS.truncate:
      x = x[1:]

    for i, sim_data in enumerate(data):
      sim_data_np = np.array(sim_data)
      y = np.mean(sim_data_np, axis=0)
      stdev = np.std(sim_data_np, axis=0)
      
      if FLAGS.truncate:
        y = y[1:]
        stdev = stdev[1:]

      if FLAGS.expsro:
        ax.plot(x, y, label=names[i], color=line_colors[i], linewidth=2)
        if FLAGS.with_std:
          ax.fill_between(x, y - stdev, y + stdev, alpha=0.2, color=line_colors[i])
      else:
        ax.plot(x, y, label=names[i], linewidth=2)
        if FLAGS.with_std:
          ax.fill_between(x, y - stdev, y + stdev, alpha=0.2)

    ax.set_title("Expected Welfare Over Iterations: {}".format(FLAGS.game_name))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Expected Welfare")
    _, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, 3))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[idx] for idx in display_order],[labels[idx] for idx in display_order])
    welfare_fig.savefig(folder_path + "expected_welfare.jpg")
    plt.close()

  def graph_expected_payoff(data, folder_path):
    # Plot the expected payoff for each player on the same plot
    expected_payoff_fig, ax = plt.subplots()
    for player_index in range(num_players):
      ax.plot([ind for ind in range(len(data[player_index]))],
        data[player_index],
        label= "{}: Player {}".format(name_of_method, player_index))
    ax.set_title("Expected Individual Payoff Over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Expected Individual Utility")
    _, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, 3))
    ax.legend()
    expected_payoff_fig.savefig(folder_path + "expected_payoff_individual.jpg")
    plt.close()

  def get_color_gradient_for_ex2psro(names):
    # Returns a list of colors to give to ex2psro lines
    color_result = [None for _ in names]
    from matplotlib import cm 
    colors_used = cm.get_cmap("viridis", len(names)).colors 
    ex2psro_names = []
    has_vanilla = False
    for i, name in enumerate(names):
      if "vanilla" in name.lower():
        color_result[i] = colors_used[0]
        has_vanilla = True 
      else:
        ex2psro_names.append(name) 
    
    def get_exponent(name):
      return -int(name.split('-')[1])  # make it negative for sorting purposes. Higher exponent means less regularization
    
    def get_weight(name):
      return int(name.split('e')[0])
    
    ex2psro_colors = colors_used[1:] if has_vanilla else colors_used # [1:4]

    # below is when we sort by name
    sorted_names = sorted(ex2psro_names, key=lambda x: x)
    
    # below is when we sort by rrd regularization 
    # sorted_names = sorted(ex2psro_names, key=lambda x: x[-1])

    # below is when we sort by regularization weight 
    # sorted_names = sorted(ex2psro_names, key=lambda x: (get_exponent(x), get_weight(x)))
    
    display_order = [None for _ in names]

    for i, name in enumerate(names):
      if name in sorted_names:
        index_in_sort = sorted_names.index(name)
        if "ex2" in name.lower():
          color_result[i] = "orangered"  # manually set ex2psro to red
        else:
          color_result[i] = ex2psro_colors[index_in_sort]
        display_order[index_in_sort + (1 if has_vanilla else 0)] = i
      else:
        # Vanilla 
        display_order[0] = i
    return color_result, display_order

  def graph_regret(data, names, folder_path):
    #Plot the individual regret values for each player on the same plot
    regret_fig, ax = plt.subplots()
    line_colors, display_order = get_color_gradient_for_ex2psro(names) if FLAGS.expsro else None
    for i, sim_data in enumerate(data):
      sim_data_np = np.array(sim_data)
      for player_index in range(num_players):
        y = []
        stdev = []
        for ind in range(len(sim_data[0][0])):
          y.append(np.mean(sim_data_np[:, player_index, ind]))
          stdev.append(np.std(sim_data_np[:, player_index, ind], axis=0))
          
        x = [ind + 1 for ind in range(len(sim_data[player_index][0]))]
        
        if FLAGS.truncate:
          x = x[3:]
          y = y[3:]
          stdev = stdev[3:]
        if FLAGS.expsro:
          ax.plot(x, y,label=names[i], color=line_colors[i], linewidth=2)
          if FLAGS.with_std:
            ax.fill_between(x, np.array(y) - np.array(stdev), np.array(y) + np.array(stdev), alpha=0.2, color=line_colors[i])
        else:
          ax.plot(x, y,label=names[i], linewidth=2)
          if FLAGS.with_std:
            ax.fill_between(x, np.array(y) - np.array(stdev), np.array(y) + np.array(stdev), alpha=0.2)
    ax.set_title("Regret Over Iterations: {}".format(FLAGS.game_name))

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Regret")
    _, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, 3))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[idx] for idx in display_order],[labels[idx] for idx in display_order])
    regret_fig.savefig(folder_path + "regret.jpg")
    plt.close()

  def graph_regret_iter_options(data, folder_path):
    '''
      Graph regret for each player individually 
    '''
    regret_fig, ax = plt.subplots()
    for player_index in range(len(data)):
      ax.plot([ind for ind in range(len(data[player_index]))],
          data[player_index], label= "{}: Player {}".format(name_of_method, player_index))
    ax.set_title("Individual Regret Over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Regret Values")
    _, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, 3))
    ax.legend()
    regret_fig.savefig(folder_path + "regret.jpg")
    plt.close()

  TRIAL_EXP_WELFARE = []
  TRIAL_REGRET = [[] for _ in range(num_players)]

  def get_data(relative_folder_path):
    """
      Displays game analytics like regret, expected utility, player profiles, and other analytics for 2-player games.
        folder_path: defines the directory where the game data is stored.

      Note that the utility matrix corresponding to player 0 is to be read in rows,
        and player 1's should be read column-wise.
    """
    total_path = data_directory_path + relative_folder_path
    all_files = sorted([f for f in os.listdir(total_path) if os.path.isfile(os.path.join(total_path, f))])
    print("ALLFILES", all_files)
    max_social_welfare_over_iterations = []
    aggregated_kl_values = [[] for _ in range(num_players)]
    expected_payoff_individual_players = [[] for _ in range(num_players)]
    expected_welfare = []
    iterations = len(all_files) if (not FLAGS.expsro) or len(all_files) < 30 else 30
    regret_individuals = [[] for _ in range(num_players)]
    player_profile_history = [[np.array([1])] for _ in range(num_players)]
    trial_graph_path = os.getcwd() + save_graph_path + relative_folder_path + "/"
    for i in range(iterations):
      print("ITERATION {}:".format(i))
      # save_folder_path + specific_string.format(i)
      save_data_path = [file for file in all_files if "iteration_{}.".format(i) in file][0]  # get the first one (cuz there's some...weird overlap ig right now)
      print("FILE", save_data_path)
      save_data_path = data_directory_path + relative_folder_path + "/" + save_data_path
      with open(save_data_path, "rb") as npy_file:
          array_list = np.load(npy_file, allow_pickle=True)
      meta_probabilities, utilities, training_returns, training_regret_returns, ppo_training_data, pure_br_returns = array_list
      if graph_mode:
        if training_returns:
          graph_training_returns(training_returns, i, trial_graph_path)
        if training_regret_returns:
          graph_regret_training_returns(training_regret_returns, i, trial_graph_path)
      # meta_probabilities was vstacked at first dimension for each player
      # utilities were vstacked at the first dimension for each player as well
      list_of_meta_probabilities = [meta_probabilities[i] for i in range(meta_probabilities.shape[0])]
      for num_player in range(num_players):
        if i == 0:
          expected_payoff_individual_players[num_player].append(utilities[num_player][0][0])
        player_profile_history[num_player].append(list_of_meta_probabilities[num_player])
        #print(utilities[num_player])
        #input(list_of_meta_probabilities)
        expected_payoff_individual = _partial_multi_dot(utilities[num_player], list_of_meta_probabilities, num_player)
        # print("current", expected_payoff_individual)
        # if i == 2:
        #   input()
        #expected_payoff_individual = _partial_multi_dot(utilities[num_player][:, :len(player_prof_history_temp[i]) - 1], player_prof_history_temp[i], num_player)
        player_profile = list_of_meta_probabilities[num_player]
        
        expected_utility_individual = np.dot(player_profile, expected_payoff_individual)
        expected_payoff_individual_players[num_player].append(expected_utility_individual)
        # ppo_training_data = [kl, entropy, actor-loss, value-loss]
        aggregated_kl_values[num_player].append(ppo_training_data[num_player][0])

        if i == (iterations - 1): 
          # The iterations allowed as options for regret (for allowing unregularized regret for ExPSRO)
          regret_iter_options = iterations
          br_iteration = 0
          if FLAGS.expsro:
            # works only if all_files is sorted and the last file in the directory is the last iteration.
            # this assumption should be true most of the time. 
            save_path = data_directory_path + relative_folder_path + "/" + all_files[-1]
            # Account for unregularized regret
            print("FILE LOADED: ", save_path)
            with open(save_path, "rb") as npy_file:
              unregularized_regret_array_list = np.load(npy_file, allow_pickle=True)
            unreg_meta_probabilities, unreg_utilities, _, _, _, _ = unregularized_regret_array_list

            # NOTE: Inserted code to replace the unreg_utilities with utilities CORRECTLY
            # This is a fail-safe mechanism just in case later iterations save data that is inconsistent with earlier ones

            for p_failsafe in range(len(utilities)):
              for r_failsafe in range(len(utilities[0])):
                for c_failsafe in range(len(utilities[0][0])):
                  unreg_utilities[p_failsafe][r_failsafe][c_failsafe] = utilities[p_failsafe][r_failsafe][c_failsafe]


            regret_iter_options = len(unreg_meta_probabilities[0])
          else:
            unreg_utilities = utilities
          for j in range(0, iterations):
            #First index in player_profile_history is iteration 1's profile
            prev_player_profile = player_profile_history[num_player][j]
            best_response_expected_payoff = 0
            for k in range(0, regret_iter_options):
              #Calculate regret for all previous iterations
              #Row player
              if num_player == 0:
                # Row corresponding to best_response utilities
                best_response_payoffs = unreg_utilities[num_player][k]
              else:
                #if symmetric, never reaches here. num_player = 0 always
                best_response_payoffs = unreg_utilities[num_player][:, k]
              best_response_trunc = best_response_payoffs[:len(prev_player_profile)]
              if np.dot(prev_player_profile, best_response_trunc) > best_response_expected_payoff:
                br_iteration = k
              best_response_expected_payoff = max(np.dot(prev_player_profile, best_response_trunc), best_response_expected_payoff)
            #regret_individuals[num_player].append(max(max(best_response_expected_payoff - expected_payoff_individual_players[num_player][i - 1], pure_br_returns[num_player] - expected_payoff_individual_players[num_player][i - 1]), 0))
            regret_individuals[num_player].append(best_response_expected_payoff - expected_payoff_individual_players[num_player][j])
            print("DEBUGGING. BR Iteration: ", br_iteration)
            TRIAL_REGRET[num_player] = regret_individuals[num_player]
          print("Individual regret vector: ", regret_individuals[num_player])
          print("Last elements of regret vector: ", regret_individuals[num_player][-5:])

        #PPO Training Returns
        #KL Divergence not used 
        #print("KL Divergence Data Individual {}: ".format(num_player), ppo_training_data[num_player][0])


      social_welfare = np.sum(utilities, axis=0)
      max_social_welfare_over_iterations.append(np.max(social_welfare))

      # # num_players is 1 if symmetric
      expected_welfare_iteration = 0
      joint_profile = []
      for val in list_of_meta_probabilities[0]:
        for val2 in list_of_meta_probabilities[1]:
          joint_profile.append(val * val2)
      # for val in list_of_meta_probabilities[0]:
      #   for val2 in list_of_meta_probabilities[1]:
      #     joint_profile.append(val * val2)
      flattened_social_welfare = np.array(social_welfare).flatten()
      expected_welfare_iteration = np.dot(flattened_social_welfare, joint_profile)
      expected_welfare_iteration = 0
      for i in range(num_players):
        expected_welfare_player = _partial_multi_dot(social_welfare, list_of_meta_probabilities, i)
        expected_welfare_iteration += np.dot(list_of_meta_probabilities[i], expected_welfare_player)
  
      print("Expected welfare: ", expected_welfare_iteration)

      TRIAL_EXP_WELFARE.append(expected_welfare_iteration)
      #Readability
      print()

    #Graphing
    if graph_mode:
      if not os.path.exists(trial_graph_path):
        os.makedirs(trial_graph_path)
      #graph_max_welfare(max_social_welfare_over_iterations, trial_graph_path)
      graph_expected_payoff(expected_payoff_individual_players, trial_graph_path)
      graph_kl(aggregated_kl_values, trial_graph_path)
      graph_regret_iter_options(regret_individuals, trial_graph_path)
    
    return utilities, meta_probabilities

  def _rrd_sims(meta_games):
    NUM_ITER = 50
    import random
    max_welfare = 0
    max_welfare_profile = []
    for k in range(NUM_ITER):
      random_nums = [np.array([random.randint(1,5) for _ in range(len(meta_games[0]))]) for _ in range(FLAGS.n_players)]
      random_profile = [player_profile / np.sum(player_profile) for player_profile in random_nums]
      #prd_dt default = 1e-3 (0.001)
      prd_profile = projected_replicator_dynamics.regularized_replicator_dynamics(
        meta_games,regret_lambda=0.01,
        prd_initial_strategies=random_profile, prd_dt=1e-3, symmetric=is_symmetric, prd_iterations=int(1e5))
      
      print("Finished iteration {}".format(k))

      combined_profile = []
      for prob in prd_profile[0]:
        for p2_prob in prd_profile[1]:
          combined_profile.append(prob * p2_prob)
      welfare = np.mean([np.dot(combined_profile, np.array(meta_games[0]).flatten()), np.dot(combined_profile, np.array(meta_games[1]).flatten())])
      if welfare > max_welfare:
        max_welfare_profile = prd_profile
        max_welfare = welfare
        print("New max welfare: ", max_welfare)
    return max_welfare_profile, max_welfare

  if is_symmetric:
    print("GAME IS SYMMETRIC. ONLY 1 SET OF RETURNS\n")

  utilities = []
  meta = []
  AGGREGATE_WELFARE = []
  AGGREGATE_REGRET = []
  AGGREGATE_METHOD_NAMES = []
  overall_start_time = time.time()
  indices_of_jobs_needed = [None for i in range(120)]

  for i, epsiode_data_directory in enumerate(sorted(os.listdir(data_directory_path))):
    SIM_WELFARE = []
    SIM_REGRET = []
    print("------------------------------FILE {}-------------------------------".format(epsiode_data_directory))
    # if "devil" not in epsiode_data_directory:
    #   continue
    for j, trial_data_directory in enumerate(sorted(os.listdir(data_directory_path + "/" + epsiode_data_directory))):
      print(trial_data_directory)
      print("TRIAL {}:\n".format(j))
      #Do welfare prd once after all trials

      utilities, meta = get_data(epsiode_data_directory + "/" + trial_data_directory)
      
      # This is my code for analyzing welfare. I've commented it out for you :)
      """
      job_id = int(trial_data_directory.split('_')[1][1:-1]) * 10 + int(trial_data_directory.split('_')[-1])

      print("Length: ", len(utilities[0]), len(indices_of_jobs_needed), job_id)
      if len(utilities[0]) > 0 and job_id < 120 and job_id >= 0: # < 75: # 
        print("Length of the EXP_WELFARE: ", len(TRIAL_EXP_WELFARE))
        print("Ending expected welfare: ", TRIAL_EXP_WELFARE[-1], len(TRIAL_EXP_WELFARE))
        indices_of_jobs_needed[job_id - 0] = TRIAL_EXP_WELFARE[-1]
      """
      
      SIM_WELFARE.append(TRIAL_EXP_WELFARE)
      SIM_REGRET.append(TRIAL_REGRET)

      TRIAL_EXP_WELFARE = []
      TRIAL_REGRET = [[] for _ in range(num_players)]
      print("Calculating alternate RRD Welfares")
      start_time = time.time()
      
      if "vanilla" not in epsiode_data_directory.lower():
        if True: # job_id not in [75, 76, 77, 78, 79]:
          continue
        print("rrd sims for job id: ", job_id)
        _, max_welfare_sims = _rrd_sims([utilities[0][:30, :30], utilities[1][:30, :30]])
        # print(_rrd_sims([utilities[0], utilities[1]]))
        print("Max welfare from sims: ", max_welfare_sims, 2*max_welfare_sims)
        indices_of_jobs_needed[job_id] = max(indices_of_jobs_needed[job_id], 2*max_welfare_sims)
        print("Individual rrd runtime in seconds: ", time.time() - start_time)
      
    AGGREGATE_WELFARE.append(SIM_WELFARE)
    AGGREGATE_REGRET.append(SIM_REGRET)  
    AGGREGATE_METHOD_NAMES.append(epsiode_data_directory)
  print("Comma separated welfare numbers: ", indices_of_jobs_needed)
  graph_expected_welfare(AGGREGATE_WELFARE, AGGREGATE_METHOD_NAMES, os.getcwd() + save_graph_path)

  graph_regret(AGGREGATE_REGRET, AGGREGATE_METHOD_NAMES, os.getcwd() + save_graph_path)
  print("TIME OVERALL: ", time.time() - overall_start_time)
  
if __name__ == "__main__":
  app.run(main)
