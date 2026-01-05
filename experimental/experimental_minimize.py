import numpy as np
from experimental_data_utils import ExperimentalData
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#configuration
freqs = np.logspace(np.log10(40), 6, 201) #40Hz to 1MHZ in 201 points
filename = './data/ICE.csv'
target_data = ExperimentalData(filename, freqs)
