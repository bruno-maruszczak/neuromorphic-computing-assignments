import numpy as np
import matplotlib.pyplot as plt


simulation_time = 1.0
time_step = 0.001
time_array = np.arange(0, simulation_time, time_step)

a = 0.01
b = 0.2
c = -65.0
d = 8.0

def run_simulation(I_value):
    V = -70.0   # Initial membrane potential in [mV]
    u = b * V
    I = np.zeros_like(time_array)
    I[(time_array >= 0.2) & (time_array < 0.6)] = I_value

    V_values = []
    spikes = 0
    for t in time_array:
        V_values.append(V)
        V0 = V

        if V >= 30:
            spikes += 1
            V = c
            u += d
        else:
            V += (0.04 * V0**2 + 5 * V0 + 140 - u + I[int(t / time_step)]) / 2 
            V += (0.04 * V**2 + 5 * V + 140 - u + I[int(t / time_step)]) / 2
            u += (a * (b * V0 - u))
            if V >= 30:
                V = 30

    return V_values, spikes, I

# Current range for simulations
current_range = np.arange(0, 51, 0.01)  # from 0 to 50 in steps of 0.01 [pA]
spike_counts = []


for I_value in current_range:
    _, spikes, _ = run_simulation(I_value)
    spike_counts.append(spikes)


plt.figure(figsize=(12, 7))
plt.plot(current_range, spike_counts, color='g', alpha=0.7)
plt.title('Number of Spikes vs Current')
plt.xlabel('Injected Current (pA)')
plt.ylabel('Number of Spikes')
plt.grid()

plt.tight_layout()
plt.show()