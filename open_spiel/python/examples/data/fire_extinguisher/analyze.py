

import numpy as np
from absl import app
from absl import flags
import os

import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

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

  data_directory_paths = ["open_spiel/python/examples/data/fire_extinguisher/imitation_game_data/"]
  iterations = len(os.listdir(data_directory_paths[0]))
  num_players = 2
  corresponding_dictionaries = []
  name_of_method = ["Fire Extinguisher"]
  save_graph_path = "open_spiel/python/examples/data/fire_extinguisher/output_plots/"
  graphPathExists = os.path.exists(save_graph_path)
  if not graphPathExists:
    # Create directory
    os.makedirs(save_graph_path)
    print("Graph directory created.")

  def graph_training_returns(training_returns, num_iteration, average_grouping = 50):
    """
      Graph training returns for BR training
        training_returns: 2D array with n arrays containing training returns for players
        num_iteration: # of current iteration
        average_grouping: Size of episode grouping (for cleaner graphing purposes)
        axis_normalization: Normalizes y-axis scale across the iterations for easier trend comparisons
    """
    grouped_rets = []
    for i in range(len(training_returns)):
      grouped_rets.append([])
      for j in range(0, len(training_returns[i]), average_grouping):
        grouped_rets[i].append(np.mean(training_returns[i][j: j + average_grouping]))
    for i in range(num_players):
      training_returns_fig, ax = plt.subplots()
      ax.plot(np.arange(0, len(grouped_rets[i])), grouped_rets[i])
      ax.set_title("BR Training Returns Player {}".format(i + 1))
      ax.set_xlabel("Iteration (Grouped by {} episodes)".format(average_grouping))
      ax.set_ylabel("Expected Returns")
      training_returns_fig.savefig(save_graph_path + "training_returns/" + "iteration_{}_player_{}.jpg".format(num_iteration, i+1))


  def get_data(folder_path):
    """
      Displays game analytics like regret, expected utility, player profiles, and other analytics for 2-player games.
        folder_path: defines the directory where the game data is stored.
      
      Note that the utility matrix corresponding to player 0 is to be read in rows, 
        and player 1's should be read column-wise.  
    """
    max_social_welfare_over_iterations = []
    expected_payoff_individual_players = [[], []]
    expected_welfare = []
    all_files = os.listdir(folder_path)
    regret_individuals = [[],[]]
    player_profile_history = [[], []]
    average_regret = []
    #We add best response and an exploration strategy's utility every iteration. Specifies index of best response utility. 
    print("Folder path: ", folder_path)
    for i in range(iterations):
      # save_folder_path + specific_string.format(i)
      save_data_path = [file for file in all_files if "iteration_{}".format(i) in file][0]
      save_data_path = folder_path + save_data_path
      print("Save_data_path", save_data_path)
      with open(save_data_path, "rb") as npy_file:
          array_list = np.load(npy_file, allow_pickle=True)
      #print(array_list)
      meta_probabilities, utilities, training_returns = array_list
      if len(array_list[-1]) > 0:
        graph_training_returns(training_returns, i)
      print("Utilities:", utilities)
      # meta_probabilities was vstacked at first dimension for each player
      # utilities were vstacked at the first dimension for each player as well
      list_of_meta_probabilities = [meta_probabilities[i] for i in range(meta_probabilities.shape[0])]
      for num_player in range(len(expected_payoff_individual_players)):
        player_profile_history[num_player].append(list_of_meta_probabilities[num_player])
        expected_payoff_vector = _partial_multi_dot(utilities[num_player], list_of_meta_probabilities, num_player)
        print("Expected payoff individual {}: ".format(num_player), expected_payoff_vector)
        player_profile = list_of_meta_probabilities[num_player]
        print("Player {} profile".format(num_player), list_of_meta_probabilities[num_player])
        expected_utility = np.dot(player_profile, expected_payoff_vector)
        expected_payoff_individual_players[num_player].append(expected_utility)
        print("Expected utility individual {}: ".format(num_player), expected_utility)
        
        if i > 0:
          #TODO: Add regret calculations for consensus policy iterations (i.e. odd iterations)
          #Even iterations add a BR, so only calc regret for those iterations
          if i % 2 == 0:
            #Row player 
            if num_player == 0:
              # Row corresponding to best_response utilities
              best_response_payoffs = utilities[num_player][-1]
            else:
              # Column corresponding to best_response utilities
              best_response_payoffs = utilities[num_player][:, -1]
            prev_player_profile = player_profile_history[num_player][i-1]
            best_response_trunc = best_response_payoffs[:len(prev_player_profile)]
            best_response_expected_payoff = np.dot(prev_player_profile, best_response_trunc)
            #Truncated to the length of the previous player profile
            print("Best response utilities truncated: ", best_response_trunc)
            #Dotted utilities with player profile
            print("Best response payoff: ", best_response_expected_payoff)
            #Previous iteration expected payoff
            print("Previous expected payoff: ", expected_payoff_individual_players[num_player][i - 1])
            regret_individuals[num_player].append(best_response_expected_payoff - expected_payoff_individual_players[num_player][i - 1])
            print("Individual regret vector: ", regret_individuals)
          else:
            print("NO REGRET CALCULATION FOR ODD ITERATIONS.")
          #Gap for readability
          print()

      social_welfare = np.sum(utilities, axis=0)
      max_social_welfare_over_iterations.append(np.max(social_welfare))

      expected_welfare_player_0 = _partial_multi_dot(social_welfare, list_of_meta_probabilities, 0)
      expected_welfare_iteration = np.dot(list_of_meta_probabilities[0], expected_welfare_player_0)
      expected_welfare.append(expected_welfare_iteration)
    
    #Average regret of both players in each iteration 
    for i in range(len(regret_individuals[0])):
      avg_regret_iter = 0
      for num_player in range(num_players):
        avg_regret_iter += regret_individuals[num_player][i]
      average_regret.append(avg_regret_iter / num_players)
    print("Average regret: ", average_regret)
    print("Max welfare:", max_social_welfare_over_iterations)
    return max_social_welfare_over_iterations, expected_payoff_individual_players, expected_welfare, regret_individuals

  #Data_directory_paths - in the current case, we only pull data from one directory but can be expanded to hold more.
  for i, path in enumerate(data_directory_paths):
    max_social_welfare, expected_payoff_individual, expected_welfare, regret_individuals = get_data(path)
    corresponding_dictionaries.append({})
    corresponding_dictionaries[i]["max_social_welfare"] = max_social_welfare
    corresponding_dictionaries[i]["expected_payoff_individual"] = expected_payoff_individual
    corresponding_dictionaries[i]["expected_welfare"] = expected_welfare
    corresponding_dictionaries[i]["regret_individuals"] = regret_individuals

  print("Analysis summary: ", corresponding_dictionaries)

  #Graph the max social welfare over iterations
  welfare_fig, ax = plt.subplots()
  for i, curr_dict in enumerate(corresponding_dictionaries):
    #Optional TODO: Fill in label with name of data
    ax.scatter(x=[ind + 1 for ind in range(len(curr_dict["max_social_welfare"]))], 
      y=curr_dict["max_social_welfare"], 
      label="{}: max welfare".format(name_of_method[i]))
  ax.set_title("Max Welfare Over Iterations")
  ax.set_xlabel("Iteration")
  ax.set_ylabel("Welfare")
  start, end = ax.get_xlim()
  ax.xaxis.set_ticks(np.arange(0, end, 1))
  welfare_fig.savefig(save_graph_path + "max_welfare.jpg")

  # Graph the expected welfare over iterations
  welfare_fig, ax = plt.subplots()
  for i, curr_dict in enumerate(corresponding_dictionaries):
    ax.scatter(x=[ind + 1 for ind in range(len(curr_dict["expected_welfare"]))], 
      y=curr_dict["expected_welfare"], label="{}: expected welfare".format(name_of_method[i]))
  ax.set_title("Expected Welfare Over Iterations")
  ax.set_xlabel("Iteration")
  ax.set_ylabel("Welfare")
  start, end = ax.get_xlim()
  ax.xaxis.set_ticks(np.arange(0, end, 1))
  welfare_fig.savefig(save_graph_path + "expected_welfare.jpg")

  # Plot the expected payoff for each player on the same plot
  expected_payoff_fig, ax = plt.subplots()
  for i, curr_dict in enumerate(corresponding_dictionaries):
    for player_index in range(num_players):
      ax.scatter(x=[ind + 1 for ind in range(len(curr_dict["expected_payoff_individual"][0]))], 
        y=curr_dict["expected_payoff_individual"][player_index], 
        label= "{}: Player {}".format(name_of_method[i], player_index))
  ax.set_title("Expected Individual Payoff Over Iterations")
  ax.set_xlabel("Iteration")
  ax.set_ylabel("Expected Individual Utility")
  start, end = ax.get_xlim()
  ax.xaxis.set_ticks(np.arange(0, end, 1))
  ax.legend()
  expected_payoff_fig.savefig(save_graph_path + "expected_payoff.jpg")

  #Plot the individual regret values for each player on the same plot
  regret_fig, ax = plt.subplots()
  for i, curr_dict in enumerate(corresponding_dictionaries):
    for player_index in range(len(regret_individuals)):
        #TODO: Fix when we figure out regret calculations for odd iterations
        ax.scatter(x=[ind * 2 + 2 for ind in range(len(curr_dict["regret_individuals"][0]))],
          y=regret_individuals[player_index], 
          label="{}: Player {}".format(name_of_method[i], player_index))
  ax.set_title("Individual Regret Over Iterations")
  ax.set_xlabel("Iteration")
  ax.set_ylabel("Regret Values")
  start, end = ax.get_xlim()
  ax.xaxis.set_ticks(np.arange(0, end, 1))
  ax.legend()
  regret_fig.savefig(save_graph_path + "regret.jpg")

if __name__ == "__main__":
  app.run(main)
