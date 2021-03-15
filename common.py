""" Common functionality and modules used by other modules """

import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
