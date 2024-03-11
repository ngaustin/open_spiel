"""
This is the main offline PSRO code runner. It will combine every combine within this folder.
"""

import time
from datetime import datetime

from absl import app
from absl import flags
import numpy as np
import os

# pylint: disable=g-bad-import-order
import pyspiel
import tensorflow.compat.v1 as tf
import sys
# pylint: enable=g-bad-import-order

FLAGS = flags.FLAGS 


# Policy Training


# Policy Evaluation 


# Target Extraction


# Game Specific




class OfflinePSRORunner:
    def __init__(self):
        return 



def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print("\n\n")
    np.random.seed(FLAGS.seed)


if __name__ == "__main__":
  app.run(main)

