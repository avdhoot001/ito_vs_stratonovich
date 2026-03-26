

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 2.0
dt = 1e-3
N = int(T/dt) + 1
M = 10000
t = np.linspace(0, T, N)

times_to_plot = [0.2, 0.5, 1.0, 2.0]
indices = [int(tt/dt) for tt in times_to_plot]

# Initialize
x_B1 = np.ones((M, N))
x_B2 = np.ones((M, N))

# Simulation
for i in range(N-1):
    dW = np.sqrt(dt) * np.random.randn(M)

    x_B1[:, i+1] = x_B1[:, i] + x_B1[:, i] * dW
    x_B2[:, i+1] = x_B2[:, i] + 0.5 * x_B2[:, i] * dt + x_B2[:, i] * dW

# PDFs
def lognormal_pdf(x, mu, var):
    return (1/(x*np.sqrt(2*np.pi*var))) * np.exp(-(np.log(x)-mu)**2/(2*var))

def normal_pdf(x, mu, var):
    return (1/np.sqrt(2*np.pi*var)) * np.exp(-(x-mu)**2/(2*var))


# ===== Plot both lognormal (x) and normal (log x) =====
for idx, tt in zip(indices, times_to_plot):

    data_B1 = x_B1[:, idx]
    data_B2 = x_B2[:, idx]

    # ===============================
    # 🔹 1. LOGNORMAL (x)
    # ===============================
    plt.figure()

    bins = np.linspace(0, np.percentile(data_B2, 99), 200)

    plt.hist(data_B1, bins=bins, density=True, alpha=0.5, label="B1 Numerical")
    plt.hist(data_B2, bins=bins, density=True, alpha=0.5, label="B2 Numerical")

    x_vals = np.linspace(1e-4, np.percentile(data_B2, 99), 500)

    pdf_B1 = lognormal_pdf(x_vals, -tt/2, tt)
    pdf_B2 = lognormal_pdf(x_vals, 0, tt)

    plt.plot(x_vals, pdf_B1, 'b--', label="B1 Analytical")
    plt.plot(x_vals, pdf_B2, 'r--', label="B2 Analytical")

    plt.title(f"LOGNORMAL PDF (x) at t = {tt}")
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.legend()
    plt.grid()
    plt.show()


    # ===============================
    # 🔹 2. NORMAL (log x)
    # ===============================
    plt.figure()

    log_B1 = np.log(data_B1)
    log_B2 = np.log(data_B2)

    bins = np.linspace(np.min(log_B1), np.max(log_B2), 200)

    plt.hist(log_B1, bins=bins, density=True, alpha=0.5, label="B1 Numerical")
    plt.hist(log_B2, bins=bins, density=True, alpha=0.5, label="B2 Numerical")

    x_vals = np.linspace(np.min(log_B1), np.max(log_B2), 500)

    pdf_B1 = normal_pdf(x_vals, -tt/2, tt)
    pdf_B2 = normal_pdf(x_vals, 0, tt)

    plt.plot(x_vals, pdf_B1, 'b--', label="B1 Analytical")
    plt.plot(x_vals, pdf_B2, 'r--', label="B2 Analytical")

    plt.title(f"NORMAL PDF (log x) at t = {tt}")
    plt.xlabel("log(x)")
    plt.ylabel("p(log x)")
    plt.legend()
    plt.grid()
    plt.show()