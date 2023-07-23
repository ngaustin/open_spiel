

import numpy as np
from absl import app
from absl import flags
import os

import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

flags.DEFINE_string("game_data_path",  "", "Directory for retrieving game data. Assuming iteration structure.")
flags.DEFINE_string("save_graph_path",  "", "Directory for saving graphs. Directory will be created if doesn't exist.")
flags.DEFINE_integer("n_players", 2, "The number of players.")
flags.DEFINE_string("game_name", "", "Name of simulation.")
flags.DEFINE_boolean("graph_mode", True, "Will/won't graph.")
flags.DEFINE_boolean("is_symmetric", True, "Symmetric game?")

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
  # For truncating printed training rets and ppo rets for readability
  TRUNCATED_MODE = True
  NUM_TRUNCATED_VALS = 50

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
    ax.xaxis.set_ticks(np.arange(0, end, 1))
    welfare_fig.savefig(folder_path + "max_welfare.jpg")
    plt.close()

  def graph_expected_welfare(data, folder_path):
    # Graph the expected welfare over iterations
    welfare_fig, ax = plt.subplots()
    ax.scatter(x=[ind for ind in range(len(data))],
      y=data, label="{}: expected welfare".format(name_of_method))
    ax.set_title("Expected Welfare Over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Welfare")
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, 1))
    welfare_fig.savefig(folder_path + "expected_welfare.jpg")
    plt.close()

  def graph_expected_payoff(data, folder_path):
    # Plot the expected payoff for each player on the same plot
    expected_payoff_fig, ax = plt.subplots()
    for player_index in range(num_players):
      ax.scatter(x=[ind for ind in range(len(data[player_index]))],
        y=data[player_index],
        label= "{}: Player {}".format(name_of_method, player_index))
    ax.set_title("Expected Individual Payoff Over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Expected Individual Utility")
    _, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, 1))
    ax.legend()
    expected_payoff_fig.savefig(folder_path + "expected_payoff.jpg")
    plt.close()

  def graph_regret(data, folder_path):
    #Plot the individual regret values for each player on the same plot
    regret_fig, ax = plt.subplots()
    for player_index in range(len(data)):
        ax.plot([ind + 1 for ind in range(len(data[player_index]))],
          data[player_index],
          label= "{}: Player {}".format(name_of_method, player_index))
    ax.set_title("Individual Regret Over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Regret Values")
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, end, 1))
    ax.legend()
    regret_fig.savefig(folder_path + "regret.jpg")
    plt.close()

  def get_data(relative_folder_path):
    """
      Displays game analytics like regret, expected utility, player profiles, and other analytics for 2-player games.
        folder_path: defines the directory where the game data is stored.

      Note that the utility matrix corresponding to player 0 is to be read in rows,
        and player 1's should be read column-wise.
    """
    all_files = os.listdir(data_directory_path + relative_folder_path + "/")
    max_social_welfare_over_iterations = []
    aggregated_kl_values = [[],[]] if not is_symmetric else [[]]
    expected_payoff_individual_players = [[], []] if not is_symmetric else [[]]
    expected_welfare = []
    iterations = len(all_files)
    regret_individuals = [[],[]] if not is_symmetric else [[]]
    player_profile_history = [[], []] if not is_symmetric else [[]] 
    average_regret = []
    trial_graph_path = os.getcwd() + save_graph_path + relative_folder_path + "/"
    for i in range(iterations):
      print("ITERATION {}:".format(i))
      # save_folder_path + specific_string.format(i)
      save_data_path = [file for file in all_files if "iteration_{}.".format(i) in file][0]
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
        player_profile_history[num_player].append(list_of_meta_probabilities[num_player])
        expected_payoff_vector = _partial_multi_dot(utilities[num_player], list_of_meta_probabilities, num_player)
        player_profile = list_of_meta_probabilities[num_player]
        
        expected_utility = np.dot(player_profile, expected_payoff_vector)
        expected_payoff_individual_players[num_player].append(expected_utility)
        # ppo_training_data = [kl, entropy, actor-loss, value-loss]
        aggregated_kl_values[num_player].append(ppo_training_data[num_player][0])

        if i > 0:
          #Row player
          if num_player == 0:
            # Row corresponding to best_response utilities
            best_response_payoffs = utilities[num_player][-1]
          else:
            # Will never get here if game is symmetric
            # Column corresponding to best_response utilities
            best_response_payoffs = utilities[num_player][:, -1]
          prev_player_profile = player_profile_history[num_player][i-1]
          best_response_trunc = best_response_payoffs[:len(prev_player_profile)]
          print(prev_player_profile, best_response_trunc)
          best_response_expected_payoff = np.dot(prev_player_profile, best_response_trunc)
          regret_individuals[num_player].append(max(best_response_expected_payoff - expected_payoff_individual_players[num_player][i - 1], pure_br_returns[num_player]- expected_payoff_individual_players[num_player][i - 1]))
          print("Individual regret vector: ", regret_individuals)

        #Truncation
        if TRUNCATED_MODE:
            training_returns[num_player] = training_returns[num_player][:NUM_TRUNCATED_VALS]
            for k in range(len(ppo_training_data)):
              ppo_training_data[num_player][k] = ppo_training_data[num_player][k][:NUM_TRUNCATED_VALS]

       #PPO Training Returns
        print("KL Divergence Data Individual {}: ".format(num_player), ppo_training_data[num_player][0])


      social_welfare = np.sum(utilities, axis=0)
      max_social_welfare_over_iterations.append(np.max(social_welfare))

      expected_welfare_player_0 = _partial_multi_dot(social_welfare, list_of_meta_probabilities, 0)
      expected_welfare_iteration = np.dot(list_of_meta_probabilities[0], expected_welfare_player_0)
      expected_welfare.append(expected_welfare_iteration)
      print("Expected welfare: ", expected_welfare_iteration)
      #Readability
      print()

    #Graphing
    if graph_mode:
      if not os.path.exists(trial_graph_path):
        os.makedirs(trial_graph_path)
      graph_max_welfare(max_social_welfare_over_iterations, trial_graph_path)
      graph_expected_welfare(expected_welfare, trial_graph_path)
      graph_expected_payoff(expected_payoff_individual_players, trial_graph_path)
      graph_regret(regret_individuals, trial_graph_path)
      graph_kl(aggregated_kl_values, trial_graph_path)
      

    return max_social_welfare_over_iterations, expected_payoff_individual_players, expected_welfare, regret_individuals

  if is_symmetric:
    print("GAME IS SYMMETRIC. ONLY 1 SET OF RETURNS\n")

  for i, trial_data_directory in enumerate(os.listdir(data_directory_path)):
    print("TRIAL {}:\n".format(i))
    get_data(trial_data_directory)
  
if __name__ == "__main__":
  app.run(main)
