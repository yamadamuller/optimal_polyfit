import numpy as np
import pandas as pd
import optimization_experimental_utils

class ExperimentalData:
    def __init__(self, path:str, freqs:np.ndarray):
        self.magnitude, self.phase = self.read_csv(path)
        self.H = self.magnitude*np.exp(1j*self.phase.astype(float))
        self.freqs = freqs

    def read_csv(self, path:str):
        data_raw = pd.read_csv(path).to_numpy() #read csv and convert to numpy array
        data_raw = data_raw[:,2:] #disregard timestamp and mode
        data_raw = np.mean(data_raw, axis=0) #compute the mean over the x-axis
        cap_idx = np.arange(0, len(data_raw), 2) #indexes of the capacitance
        res_idx = np.arange(1, len(data_raw), 2) #indexes of the resistance

        return data_raw[cap_idx], data_raw[res_idx]

