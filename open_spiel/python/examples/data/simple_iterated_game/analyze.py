

import numpy as np
from absl import app
from absl import flags

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

def main(argv):
  """ Load and analyze a list of numpy arrays representing:
        [num_players, S] shape matrix of meta_probabilities (where S is the number of policies so far)
        [num_players, num_states, num_actions, S] shape matrix representing policies
        [num_players, S, S] shape matrix representing utilities (U)

     All three of these are contained in each of the interation .npy files for each iteration
  """
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  save_folder_path = "other_player_action_conditioned_simple_game/"
  specific_string = "iteration_{}.npy"
  iterations = 5
  num_players = 2
  single_state = False

  make_payoff_matrix = True  # heatmap for every iteration
  make_policy_matrix = True  #
  make_meta_probabilities_matrix = True  # bar graph for each player for each iteration

  for i in range(iterations):
    save_data_path = save_folder_path + specific_string.format(i)
    with open(save_data_path, "rb") as npy_file:
      array_list = np.load(npy_file, allow_pickle=True)
    meta_probabilities, policies, utilities = array_list

    # Make a plot with 4 subplots: For each player (2), make 1 with meta probabilities and 1 with policies
    subplot_titles = ["Player 0 Profile", "Player 1 Profile", "Player 0 Strategy Set", "Player 1 Strategy Set"] if single_state else \
                      ["Player 0 Profile", "Player 1 Profile", "Player 0 Policy #{}".format(i+1), "Player 1 Policy #{}".format(i+1)]
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=.03, vertical_spacing=0,
                        subplot_titles=subplot_titles,
                        specs=[[{"type": "table"}, {"type": "table"}],
                              [{"type": "table"}, {"type": "table"}]])
    fig['layout'].update(height=350, width=1700, title="Summary for Iteration {}".format(i+1))

    payoff_fig = make_subplots(rows=1, cols=2, horizontal_spacing=.03, subplot_titles=["Player 0 Payoff Matrix", "Player 1 Payoff Matrix"],
                               specs=[[{"type": "image"}, {"type": "image"}]])

    for j in range(num_players):
      """ Update the iteration subplot with the meta-probabilities """
      if make_meta_probabilities_matrix:
        meta_p = np.around(meta_probabilities[j], decimals=3)
        num_strategies = meta_p.size
        policy_names = ["Policy #{}".format(str(i)) for i in range(num_strategies)]
        fig.add_trace(go.Table(header=dict(values=policy_names), cells=dict(values=meta_p)), row=1, col=(j+1))

      """ Update the iteration subplot with the policy information """
      if make_policy_matrix:
        curr_policies = np.around(policies[j], decimals=3)
        if single_state:
          curr_policies = curr_policies[0]  # assume only get one of the states
          num_strategies = curr_policies.shape[1]
          policy_names = ["Policy #{}".format(str(i)) for i in range(num_strategies)]
          headers = ["Action Index"] + policy_names
          values = curr_policies.T
          actions = np.array([[0, 1, 2]])
          values = np.vstack((actions, values))
          fig.add_trace(go.Table(header=dict(values=headers), cells=dict(values=values.tolist())), row=2, col=(j+1))
        else:
          # If we have multiple states, revise the display to be actions x states. Display one chart per iteration
          # representing the new policy added this iteration
          curr_policies = curr_policies[:, :, -1]
          # curr_policies = curr_policies.T  # num_actions by num_states
          state_names = ["Start", "Opponent Action = 0", "Opponent Action = 1", "Opponent Action = 2"]
          headers = ["Action Index"] + state_names
          actions = np.array([[0, 1, 2]])
          values = np.vstack((actions, curr_policies))
          fig.add_trace(go.Table(header=dict(values=headers), cells=dict(values=values.tolist())), row=2, col=(j+1))


      if make_payoff_matrix:
        num_strategies=utilities[j].shape[0]
        policy_names = ["Policy #{}".format(str(i)) for i in range(num_strategies)]
        payoff_fig.add_trace(go.Heatmap(z=utilities[j], x=policy_names, y=policy_names), row=1, col=(j+1))

    payoff_fig.show()
    fig.show()

  return


if __name__ == "__main__":
  app.run(main)