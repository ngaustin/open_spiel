

import numpy as np
from absl import app
from absl import flags
import os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FLAGS = flags.FLAGS

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

  save_folder_paths = ["prob_reward_100_trajectories/", "optimistic_q_100_trajectories/", "psro_normal/"]
  name_of_method = ["With Probabilistic Reward Fitting for Exploration", "With Joint Action Q-Learning for Exploration", "Normal PSRO without Exploration"]
  corresponding_dictionaries = [{}, {}, {}]
  iterations = 5
  num_players = 2


  def get_data(folder_path):
    max_social_welfare_over_iterations = []
    expected_payoff_individual_players = [[], []]
    expected_welfare = []
    regret_individual_players = [[], []]

    all_files = os.listdir(folder_path)
    max_utility_in_game_individual = 10.199999809265137

    for i in range(iterations):
      save_data_path = [file for file in all_files if "iteration_{}".format(i) in file][0]# save_folder_path + specific_string.format(i)
      save_data_path = folder_path + save_data_path
      with open(save_data_path, "rb") as npy_file:
        array_list = np.load(npy_file, allow_pickle=True)
      meta_probabilities, utilities = array_list

      # meta_probabilities was vstacked at first dimension for each player
      # utilities were vstacked at the first dimension for each player as well

      list_of_meta_probabilities = [meta_probabilities[i] for i in range(meta_probabilities.shape[0])]
      for i in range(len(expected_payoff_individual_players)):
        print("Max utility individual: ", np.max(utilities[i]))
        expected_payoff_vector = _partial_multi_dot(utilities[i], list_of_meta_probabilities, i)
        player_profile = list_of_meta_probabilities[i]
        expected_utility = np.dot(player_profile, expected_payoff_vector)
        expected_payoff_individual_players[i].append(expected_utility)

        max_expected_payoff = np.max(expected_payoff_vector)
        regret_individual_players[i].append(max_expected_payoff - expected_utility)


      social_welfare = np.sum(utilities, axis=0)
      max_social_welfare_over_iterations.append(np.max(social_welfare))

      expected_welfare_player_0 = _partial_multi_dot(social_welfare, list_of_meta_probabilities, 0)
      expected_welfare_iteration = np.dot(list_of_meta_probabilities[0], expected_welfare_player_0)
      expected_welfare.append(expected_welfare_iteration)

    return max_social_welfare_over_iterations, expected_payoff_individual_players, expected_welfare, regret_individual_players

  for i, path in enumerate(save_folder_paths):
    # print(get_data(path))
    max_social_welfare, expected_payoff_individual, expected_welfare, _ = get_data(path)
    # print(max_social_welfare)
    dict_to_edit = corresponding_dictionaries[i]
    dict_to_edit["max_social_welfare"] = max_social_welfare
    dict_to_edit["expected_payoff_individual"] = expected_payoff_individual
    dict_to_edit["expected_welfare"] = expected_welfare
    corresponding_dictionaries[i] = dict_to_edit
    print('edited dict {}'.format(i))


  # print(yes)

  # Graph the max social welfare over iterations
  print(corresponding_dictionaries)
  welfare_fig = go.Figure()
  x_axis = list(range(1, iterations + 1))
  for i in range(len(save_folder_paths)):
    welfare_fig.add_trace(go.Scatter(x=x_axis, y=corresponding_dictionaries[i]["max_social_welfare"], name=name_of_method[i], mode="lines"))
  welfare_fig.update_layout(title="Max Welfare Over Iterations", xaxis_title="Iteration", yaxis_title="Welfare")
  welfare_fig.show()

  # Graph the expected welfare over iterations
  welfare_fig = go.Figure()
  for i in range(len(save_folder_paths)):
    welfare_fig.add_trace(
      go.Scatter(x=x_axis, y=corresponding_dictionaries[i]["expected_welfare"], name=name_of_method[i], mode="lines"))
  welfare_fig.update_layout(title="Expected Welfare Over Iterations", xaxis_title="Iteration", yaxis_title="Welfare")
  welfare_fig.show()

  # Plot the expected payoff for each player on the same plot
  expected_payoff_fig = go.Figure()
  for method_index in range(len(save_folder_paths)):
    for player_index in range(num_players):
      expected_payoff_fig.add_trace(go.Scatter(x=x_axis, y=corresponding_dictionaries[method_index]["expected_payoff_individual"][player_index], name=name_of_method[method_index] + " for Player {}".format(player_index), mode="lines"))
  expected_payoff_fig.update_layout(title="Expected Individual Payoff Over Iterations", xaxis_title="Iteration", yaxis_title="Expected Individual Utility")
  expected_payoff_fig.show()

  # Plot the individual regret for each player on the same plot
  """
  regret_fig = go.Figure()
  for player_index in range(len(regret_individual_players)):
    regret_fig.add_trace(
      go.Scatter(x=list(range(1, iterations + 1)), y=regret_individual_players[player_index],
                 name="Individual Regret Player {}".format(player_index), mode="lines"))
  regret_fig.update_layout(title="Individual Regret Over Iterations", xaxis_title="Iteration",
                                    yaxis_title="Regret")
  regret_fig.show()
  """
  return


if __name__ == "__main__":
  app.run(main)