"""
Benchmark: analytic Jacobian vs central finite differences.

Shows how wall time scales with the number of resistivity parameters (layers)
for a fixed receiver array and frequency.
"""
import time

import empymod
import matplotlib.pyplot as plt
import numpy as np

from empygrad.model import dipole

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_LAYERS_LIST = [5, 10, 20, 30, 50, 75, 100, 150]
N_REPEAT = 3       # repetitions per measurement (take the minimum)
H_FD = 1e-4        # central-FD step size
N_REC = 20         # number of receivers

SRC = [0, 0, 0.1]
REC = [np.linspace(500, 5000, N_REC), np.zeros(N_REC), 0.1]
FREQ = 1.0


def build_model(n_layers):
    """Return a simple n_layers model (log-spaced depths, alternating res)."""
    depths = np.linspace(0, 2000, n_layers)
    res = np.ones(n_layers + 1)
    res[0] = 1e20                  # air
    res[1::2] = 10.0
    res[2::2] = 1.0
    return depths, res


def time_analytic(depths, res):
    t0 = time.perf_counter()
    dipole(src=SRC, rec=REC, depth=depths, res=res,
           freqtime=FREQ, ab=11, verb=0, jac='res')
    return time.perf_counter() - t0


def time_fd_empymod(depths, res):
    t0 = time.perf_counter()
    n = len(res)
    for k in range(n):
        res_p = res.copy(); res_p[k] *= (1 + H_FD)
        res_m = res.copy(); res_m[k] *= (1 - H_FD)
        empymod.dipole(src=SRC, rec=REC, depth=depths, res=res_p,
                       freqtime=FREQ, ab=11, verb=0)
        empymod.dipole(src=SRC, rec=REC, depth=depths, res=res_m,
                       freqtime=FREQ, ab=11, verb=0)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------
times_analytic = []
times_fd_mod = []

print(f"{'n_layers':>10}  {'analytic [s]':>14}  {'FD empymod [s]':>16}  {'FD empygrad [s]':>17}  {'speedup (grad)':>15}")
print("-" * 75)

for n in N_LAYERS_LIST:
    d, r = build_model(n)

    t_a  = min(time_analytic(d, r)     for _ in range(N_REPEAT))
    t_fm = min(time_fd_empymod(d, r)   for _ in range(N_REPEAT))

    times_analytic.append(t_a)
    times_fd_mod.append(t_fm)
    print(f"{n:>10d}  {t_a:>14.4f}  {t_fm:>16.4f}  {t_fm/t_a:>14.1f}x")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.plot(N_LAYERS_LIST, times_fd_mod,  "o-", color='olive', label="FD with empymod.dipole  (2n fwd calls)")
ax.plot(N_LAYERS_LIST, times_analytic, "s-", color='tomato', label="Analytic Jacobian (empygrad)")
ax.set_xlabel("Number of resistivity parameters")
ax.set_ylabel("Wall time (s)")
ax.set_title("Computation time vs number of layers")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
speedup_mod  = np.array(times_fd_mod)  / np.array(times_analytic)
ax.plot(N_LAYERS_LIST, speedup_mod,  "o-",color='tomato', label="FD empymod / analytic")
ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Number of resistivity parameters")
ax.set_ylabel("Speedup (FD time / analytic time)")
ax.set_title("Analytic speedup over finite differences")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig("benchmark_jac_vs_fd.png", dpi=150)
plt.show()
print("\nFigure saved to benchmark_jac_vs_fd.png")