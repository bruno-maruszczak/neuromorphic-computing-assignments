import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, dual_annealing

simulation_time=0.08
time_step=0.001
# Time sections for logic states
section_duration = simulation_time / 4
steps_per_section = int(section_duration / time_step)
time_array = np.arange(0, simulation_time, time_step)

v_low = 10  # [pA]
v_high = 20  # [pA]


def simulate_izhikevich(w1, w2):

    currents = [
        w1 * v_low + w2 * v_low,
        w1 * v_low + w2 * v_high,
        w1 * v_high + w2 * v_low,
        w1 * v_high + w2 * v_high,
    ]
    
    # Izhikevich model parameters
    a = 0.01
    b = 0.2
    c = -65.0
    d = 8.0

    V = -70.0   # Initial membrane potential in [mV]
    u = b * V
    V_values = []
    
    # Dictionary to hold delays for each logic input pair
    out = {key: None for key in ['00', '01', '10', '11']}
    


    for idx, curr in enumerate(currents):
        section_spike_time = None
        
        for step in range(steps_per_section):
            t = step * time_step
            
            I = curr
            
            V_values.append(V)
            V0 = V
            
            if V >= 30:
                if section_spike_time is None:
                    section_spike_time = t  # Record first spike time for this section
                V = c
                u += d
            else:
                V += (0.04 * V0**2 + 5 * V0 + 140 - u + I) / 2 
                V += (0.04 * V**2 + 5 * V + 140 - u + I) / 2
                u += (a * (b * V0 - u))
                if V >= 30:
                    V = 30
        
        # Store the spike time for the corresponding logic input
        out[['00', '01', '10', '11'][idx]] = section_spike_time if section_spike_time is not None else np.nan

    return out, np.array(V_values), currents

# Initial weights
w1, w2 = 1., 1.
delays, V_values, currents = simulate_izhikevich(w1, w2)
print("Output delays after initial simulation:", delays)

# Define the objective function for optimization
def objective(weights):
    w1, w2 = weights
    out, _, _ = simulate_izhikevich(w1, w2)
    return (0.01 - out['00']) ** 2 + (0.001 - out['01']) ** 2 + \
           (0.001 - out['10']) ** 2 + (0.01 - out['11']) ** 2

# Optimize weights to minimize the objective function
# result = minimize(objective, [w1, w2])
# w1_opt, w2_opt = result.x
#w1_opt, w2_opt = w1, w2
bounds = [(0, 5), (0, 5)] # Arbitrary
result = dual_annealing(objective, bounds, maxiter=10000)
w1_opt, w2_opt = result.x
delays, V_values, currents = simulate_izhikevich(w1_opt, w2_opt)
print("Output delays after optimization:", delays)


# Plotting the responses for the continuous simulation
fig, axs = plt.subplots(3, 1, figsize=(12, 12), height_ratios=(2, 1, 1))
# Membrane Voltage Plot
axs[0].plot(time_array[:len(V_values)], V_values, label='Membrane Voltage (V)', color='b')
axs[0].axhline(y=30, color='r', linestyle='--', label='Firing threshold (30 mV)')

# Draw vertical lines for spike times for each logic signal
for idx, key in enumerate(['00', '01', '10', '11']):
    spike_delay = delays[key]
    if spike_delay is not None:
        spike_time = idx * 0.02 + spike_delay
        axs[0].axvline(x=spike_time, color='orange', linestyle=':', lw=2, label=f'Spike Time for {key}')

axs[0].set_title('Izhikevich Neuron Model Responses to Logic Signals')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Voltage (mV)')
axs[0].set_ylim(-100, 50)
axs[0].legend()
axs[0].grid()


v1_values = np.concatenate([np.full(steps_per_section, (v_low if idx <= 1 else v_high)) for idx in range(4)])
axs[1].plot(time_array[:len(v1_values)], v1_values, label='Input Voltage v1', color='g')
axs[1].set_title('Input Voltage v1')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Voltage (pA)')
axs[1].grid()
axs[1].legend()

v2_values = np.concatenate([np.full(steps_per_section, (v_high if idx % 2 == 1 else v_low)) for idx in range(4)])
axs[2].plot(time_array[:len(v2_values)], v2_values, label='Input Voltage v2', color='m')
axs[2].set_title('Input Voltage v2')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Voltage (pA)')
axs[2].grid()
axs[2].legend()

plt.tight_layout()
plt.show()
