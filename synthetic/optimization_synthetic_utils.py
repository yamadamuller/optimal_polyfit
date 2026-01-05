import numpy as np
from synthetic.synthetic_data_utils import SyntheticData

class Optimizer:
    def __init__(self):
        self.k = 1e8
        self.scale = np.array([1e6, 1, 1, 1, 1, 1e6]) #scale to avoid issues with minimize

    def compute_response(self, params:np.ndarray, freq:np.ndarray):
        w = 2*np.pi*freq #Hz to rad/s
        jw = 1j*w #real to complex
        order = int(len(params[:])/2) #order of the polynomial
        poly = np.stack([np.ones_like(w), jw, jw ** 2]).T #array with [1, jw, jw^2]
        num = poly@params[order:] #b0+b1*jw+b2*jw²+...bn*jw^n
        den = poly@params[:order] #a0+a1*jw+a2*jw²+...an*jw^n
        H = num/den #frequency response

        return H, np.abs(H), np.angle(H)

    def norm_diff_cost(self, theta:np.ndarray, args:list[np.ndarray]):
        opt_data = SyntheticData(theta, args[1]) #generate the synthetic data for the candidate parameters
        return np.linalg.norm(args[0]-opt_data.H_scaled) #compare scaled responses

    def residue_sum_cost(self, theta:np.ndarray, args:list[np.ndarray]):
        opt_data = SyntheticData(theta, args[1]) #generate the synthetic data for the candidate parameters
        return np.sum(np.abs(args[0]-opt_data.H_scaled)) #sum of the residues

    def euclidean_sum_cost(self, theta:np.ndarray, args:list[np.ndarray]):
        opt_data = SyntheticData(theta, args[1]) #generate the synthetic data for the candidate parameters
        return np.sum(np.sqrt((np.abs(args[0]-opt_data.H_scaled))**2))
