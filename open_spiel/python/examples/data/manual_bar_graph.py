import numpy as np
from absl import app
from absl import flags
import os

import matplotlib.pyplot as plt
from open_spiel.python.examples.data.analyze import get_color_gradient_for_ex2psro

game_name = ["Harvest Sparse", "Harvest Dense", "Bargaining (.99)", "Bargaining (.95)"]
means = [[108.57, 129.02, 127.36, 119.94, 123.56, 126.75], 
        [119.57, 150.22, 142.04, 143.53, 128.15, 146.04],
        [11.42, 11.97, 11.53, 11.8, 11.55, 11.59],
        [11.68, 12.01, 12.03, 12.14, 11.65, 12.00]]
y_min_max = [[95, 135], [110, 155], [10.85, 12.15], [10.76, 12.28]]
std_devs = [[8.95, 7.54, 7.51, 10.1, 6.98, 6.25], 
            [3.35, 7.86, 6.11, 9.45, 4.33, 8.26], 
            [0.25, 0.46, 0.64, 0.37, 0.28, 0.45], 
            [0.54, 0.39, 0.55, 0.44, 0.47, 0.44]]
xticks = [i * .6 for i in range(6)]
labels = ["Vanilla", "Ex2psro", "MaxWelfare", "MinWelfare", "PrevBR", "Uniform"]
labels_to_colors, display_order = get_color_gradient_for_ex2psro(labels)

# labels_to_colors[i] gives the color name associated with label labels[i]
# display_order[i] states that the i-th label displayed should be labels[display_order[i]]

title = "Normalized Welfare Metrics"

fig, ax = plt.subplots(nrows=1, ncols=4)

game_index = 0
# for row in ax:
for col in ax: # in row:
    bar = col.bar(xticks, means[game_index], width=.6, label=labels, color=labels_to_colors)#, yerr=std_devs[game_index])
    col.set_xticks([])
    col.set_xlabel(game_name[game_index], fontweight='bold', fontsize=6)
    print(len(y_min_max), game_index)
    col.set_ylim(y_min_max[game_index])
    if game_index == 0:
        col.set_ylabel("Welfare")

    game_index += 1

fig.tight_layout()
import matplotlib.patches as mpatches

patches = []
for label, color in zip(labels, labels_to_colors):
    patches.append(mpatches.Patch(color=color, label=label))

fig.legend(handles=patches, loc="lower center", ncol=len(labels), fontsize="x-small")
fig.subplots_adjust(bottom=.15, top=.9, wspace=.3)
fig.set_size_inches(8, 3.5)
# fig.suptitle(title)

fig.savefig("open_spiel/python/examples/data/welfare_bars.png")
print("Something was saved?")