import numpy as np
from scipy.optimize import minimize
import synthetic_data_utils
import optimization_synthetic_utils
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

freqs = np.logspace(np.log10(40), 6, 201) #40Hz to 1MHZ in 201 points
optimizer_obj = optimization_synthetic_utils.Optimizer() #Optimizer object

#target parameters
target_params = np.array([0., 1e6, 1., 300 * optimizer_obj.k, optimizer_obj.k, 1.]) #target polynomial parameters
target_data = synthetic_data_utils.SyntheticData(target_params, freqs) #target SyntheticData object
target_cost = optimizer_obj.norm_diff_cost(target_params, [target_data.H_scaled, freqs]) #the value of the cost function at the target parameters
print(f'[Optimizer @ target] cost target val = {target_cost}')
print(f'[Optimizer @ target] params target val = {target_params.ravel()}')

#optimized candidate parameters
#norm cost function
initial_guess = np.array([0., 0.01e6, 0.01, 3 * optimizer_obj.k, 0.01 * optimizer_obj.k, 0.01]) #initial guess (1% of the target parameters)
opt = minimize(optimizer_obj.norm_diff_cost, initial_guess, args=[target_data.H_scaled, freqs], tol=1e-9) #find the optimal parameters that minimize the cost function
unscaled_opt_vals = opt.x[:, np.newaxis] / optimizer_obj.scale[:, np.newaxis] #unscale the optimal parameters
opt_data = synthetic_data_utils.SyntheticData(unscaled_opt_vals.ravel(), freqs) #optimized SyntheticData object
print(f'[Optimizer @ optimized norm] cost opt val = {opt.fun}')
print(f'[Optimizer @ optimized norm] params opt val = {unscaled_opt_vals.ravel()}')

#parameters direct in the equation
w = 2*np.pi*freqs #Hz to rad/s
jw = 1j*w #real to complex
order = int(len(unscaled_opt_vals)/2) #order of the polynomial
poly = np.stack([np.ones_like(w), jw, jw ** 2]).T #array with [1, jw, jw^2]
num = poly@unscaled_opt_vals[3:]
den = poly@unscaled_opt_vals[:3]
H = num/den

plt.figure(1)
plt.suptitle("||x-x'||")
plt.subplot(1,2,1)
leg = []
plt.loglog(target_data.freqs, target_data.magnitude_unscaled)
leg.append('Target values')
plt.loglog(opt_data.freqs, np.abs(H), linestyle='dotted')
leg.append('Optimized values')
plt.title("Magnitude Response")
plt.xlabel('Frequency [Hz]')
plt.ylabel('|H(jω)|')
plt.legend(leg)
plt.grid(which='both')

plt.subplot(1,2,2)
leg = []
plt.semilogx(target_data.freqs, target_data.phase_unscaled)
leg.append('Target values')
plt.semilogx(opt_data.freqs, np.angle(H), linestyle='dotted')
leg.append('Optimized values')
plt.title("Phase Response")
plt.xlabel('Frequency [Hz]')
plt.ylabel('∠H(jω) [rad]')
plt.legend(leg)
plt.grid(which='both')
plt.show()

