import numpy as np
import matplotlib.pyplot as plt


simulation_time = 1.0
time_step = 0.001
time_array = np.arange(0, simulation_time, time_step)

##  Izhikevich model params
# a = time scale of recovery variable u. Magnitude is proportional to recovery speed (small = slower recovery time). A typical value of 0.02
# b = sensitivity of u to the fluctuations in v. The bigger the value, the more their coupled, meaning the more there are subthreshold oscillations and low threshold spiking. A typical value of 0.2.
# b < a = saddle-node bifurcation in the resting state
# b > a = Andronov-Hopf bifurcation in the resting state
# c = after-spike reset value of v. A typical value of -65mV
# d = after-spike reset value of u. A typical value of 2
## Uncomment a set of params you want to test

# Typical
# a = 0.02
# b = 0.2
# c = -65
# d = 2.0

# Intrinscaly bursting
# a = 0.02
# b = 0.2
# c = -55.
# d = 4.0

# Regular spiking
# a = 0.02
# b = 0.2
# c = -65.
# d = 8.0

# Chattering
# a = 0.02
# b = 0.2
# c = -50.
# d = 2.0

# Matching
a = 0.01
b = 0.2
c = -65.
d = 8.0

V = -70.0   # Initial membrane potential in [mV]
u = b * V
I = np.zeros_like(time_array)  # Current values in [pA]
I[(time_array >= 0.2) & (time_array < 0.6)] = 20.


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
        # increment twice for numerical stability as in the paper introducing the model
        V += (0.04 * V0**2 + 5 * V0 + 140 - u + I[int(t / time_step)])/2 
        V += (0.04 * V**2 + 5 * V + 140 - u + I[int(t / time_step)])/2
        u += (a * (b * V0 - u))
        if V >= 30:
            V = 30
V_values = np.array(V_values)
I_values = np.array(I)


plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_array, V_values, label='Membrane Voltage (V)', color='b')
plt.title('Izhikevich Neuron Model')
plt.ylabel('Voltage (mV)')
plt.ylim(-100, 50)
plt.axhline(y=30, color='r', linestyle='--', label='Firing threshold (30 mV)')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time_array, I_values, label='Injected Current (I)', color='g')  # pA
plt.title('Injected Current')
plt.xlabel('Time (s)')
plt.ylabel('Current (pA)')
plt.axhline(y=20, color='orange', linestyle='--', label='Injected Current (20 pA)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()