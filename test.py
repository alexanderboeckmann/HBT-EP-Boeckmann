import numpy as np
import matplotlib.pyplot as plt

# Parameters
R = 6               # ohms
L = 30e-6           # henries
Vs = 0.024          # volts (24 mV)
tau = L / R         # time constant
I_inf = Vs / R      # steady-state current

# Time vector: 0 to 30 microseconds
t = np.linspace(0, 30e-6, 1000)

# Current response
iL = I_inf * (1 - np.exp(-t / tau))

# Plot
plt.figure(figsize=(6,4))
plt.plot(t * 1e6, iL * 1e3)  # convert to µs and mA
plt.title("Step Response of RL Circuit")
plt.xlabel("Time (µs)")
plt.ylabel("Inductor Current iL (mA)")
plt.grid(True)
plt.axhline(I_inf * 1e3, color='r', linestyle='--', label='Steady-state (4 mA)')
plt.legend()
plt.show()
