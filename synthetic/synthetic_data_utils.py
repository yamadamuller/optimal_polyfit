import numpy as np
import optimization_synthetic_utils


class SyntheticData:
    def __init__(self, params:np.ndarray, freqs:np.ndarray):
        self.params = params #candidate parameters of the polynomial
        self.freqs = freqs #array of the frequencies
        self.optimizer_obj = optimization_synthetic_utils.Optimizer() #Optimizer object
        self.scaled_params = params[:,np.newaxis]*self.optimizer_obj.scale[:, np.newaxis] #scale the candidate parameters
        self.H_scaled, self.magnitude_scaled, self.phase_scaled = self.optimizer_obj.compute_response(self.scaled_params, self.freqs) #compute the scaled frequency response
        self.H_unscaled, self.magnitude_unscaled, self.phase_unscaled = self.optimizer_obj.compute_response(self.params, self.freqs) #compute the unscaled frequency response