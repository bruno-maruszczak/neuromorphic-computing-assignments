import numpy as np
import matplotlib.pyplot as plt


simulation_time = 1.0
time_step = 0.001
time_array = np.arange(0, simulation_time, time_step)
I = np.ones_like(time_array)*0.
t_start = 0.2
t_end = 0.4
t_mid = (t_start + t_end)/ 2
delta = 40.
I[(time_array >= t_start) & (time_array < t_mid)] += np.linspace(0., delta, len(I[(time_array >= t_start) & (time_array < t_mid)]))
I[(time_array >= t_mid) & (time_array < t_end)] += np.linspace(delta, 0., len(I[(time_array >= t_mid) & (time_array < t_end)]))
##  Izhikevich model params
# a = time scale of recovery variable u. Magnitude is proportional to recovery speed (small = slower recovery time). A typical value of 0.02
# b = sensitivity of u to the fluctuations in v. The bigger the value, the more their coupled, meaning the more there are subthreshold oscillations and low threshold spiking. A typical value of 0.2.
# b < a = saddle-node bifurcation in the resting state
# b > a = Andronov-Hopf bifurcation in the resting state
# c = after-spike reset value of v. A typical value of -65mV
# d = after-spike reset value of u. A typical value of 2
## Uncomment a set of params you want to test

tonic_spiking = (0.02, -0.1, -65., 6.)
inhibition_induced_spiking = (-0.02, -1., -60., 8.)


def simulate_izhikevich(params):
    a, b, c, d = params

    V = -70.0   # Initial membrane potential in [mV]
    u = b * V

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
    
    return np.array(V_values)


v1 = simulate_izhikevich(tonic_spiking)
v2 = simulate_izhikevich(inhibition_induced_spiking)

fig, axs = plt.subplots(3, 1, figsize=(12, 6))

axs[0].plot(time_array, v1, color='b')
axs[0].set_title('Tonic Spiking')
axs[0].set_ylabel('Voltage (mV)')
axs[0].set_ylim(-100, 50)
axs[0].axhline(y=30, color='r', linestyle='--', label='Firing threshold (30 mV)')
axs[0].grid()


axs[1].plot(time_array, v2, color='b')
axs[1].set_title('Inhibition Induced Spiking')
axs[1].set_ylabel('Voltage (mV)')
axs[1].set_ylim(-100, 50)
axs[1].axhline(y=30, color='r', linestyle='--', label='Firing threshold (30 mV)')
axs[1].grid()



axs[2].plot(time_array, I, color='g')
axs[2].set_title('Input Current (A)')
axs[2].set_ylabel('Current (pA)')
axs[2].grid()


plt.tight_layout()
plt.show()