import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 2.0
dt = 1e-3
N = int(T/dt)
M = 5000  # number of trajectories
t = np.linspace(0, T, N)

# Initialize
x_B1 = np.ones((M, N))
x_B2 = np.ones((M, N))

# Simulation
for i in range(N-1):
    dW = np.sqrt(dt) * np.random.randn(M)

    # B1: Ito
    x_B1[:, i+1] = x_B1[:, i] + x_B1[:, i] * dW

    # B2: Stratonovich (Ito equivalent)
    x_B2[:, i+1] = x_B2[:, i] + 0.5 * x_B2[:, i] * dt + x_B2[:, i] * dW

# Compute moments
mean_B1 = np.mean(x_B1, axis=0)
mean_B2 = np.mean(x_B2, axis=0)

mean2_B1 = np.mean(x_B1**2, axis=0)
mean2_B2 = np.mean(x_B2**2, axis=0)

# Analytical
mean_B1_analytical = np.ones(N)
mean_B2_analytical = np.exp(t/2)

mean2_B1_analytical = np.exp(t)
mean2_B2_analytical = np.exp(2*t)

# Plot mean
plt.figure()
plt.plot(t, mean_B1, label="B1 Numerical")
plt.plot(t, mean_B1_analytical, '--', label="B1 Analytical")
plt.plot(t, mean_B2, label="B2 Numerical")
plt.plot(t, mean_B2_analytical, '--', label="B2 Analytical")
plt.legend()
plt.title("Mean ⟨x⟩")
plt.xlabel("t")
plt.show()

# Plot second moment
plt.figure()
plt.plot(t, mean2_B1, label="B1 Numerical")
plt.plot(t, mean2_B1_analytical, '--', label="B1 Analytical")
plt.plot(t, mean2_B2, label="B2 Numerical")
plt.plot(t, mean2_B2_analytical, '--', label="B2 Analytical")
plt.legend()
plt.title("Second Moment ⟨x²⟩")
plt.xlabel("t")
plt.show()

# PDF comparison at final time
plt.figure()
plt.hist(x_B1[:, -1], bins=100, density=True, alpha=0.5, label="B1")
plt.hist(x_B2[:, -1], bins=100, density=True, alpha=0.5, label="B2")
plt.legend()
plt.title("PDF at final time")
plt.show()