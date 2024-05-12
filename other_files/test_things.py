from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
from openmmplumed import PlumedForce
import numpy as np
from cmaes import CMA
import matplotlib.pyplot as plt
import os
import re
from bias import MolSim
from sklearn import preprocessing
import matplotlib.cm as cm
# import matplotlib.patches as patches
from matplotlib.patches import Circle


optimizer = CMA(mean=np.zeros(40**2), sigma=5, bounds=np.array([(0, 30)] * 40**2))
print(optimizer.population_size)
