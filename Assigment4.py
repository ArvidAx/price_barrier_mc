import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

np.random.seed(42)

s0 = 14
k = 15
r = 0.05
T = 0.5
B = 20
N = 1000
M = 1000
sigma = 0.1
gamma = 1
ds = 0.1


def payoff(S, k):
    return np.maximum(S - k, 0.0)


def barrier_option_price(s0, k, r, T, B, sigma, gamma, N, M):
    dt = T / M
    sqrt_dt = np.sqrt(dt)

    S = np.full(N, s0)
    knocked_out = np.zeros(N, dtype=bool)

    for _ in range(M):
        dW = np.random.normal(0, sqrt_dt, N)
        S = np.maximum(0.0, S + r * S * dt + sigma * (S ** gamma) * dW)
        knocked_out |= (S >= B)

    S[knocked_out] = 0.0
    return np.exp(-r * T) * np.mean(payoff(S, k))


def calculate_delta(s0, k, r, T, B, sigma, gamma, N, M, ds):
    dt = T / M
    sqrt_dt = np.sqrt(dt)

    S_up = np.full(N, s0 + ds)
    S_down = np.full(N, s0 - ds)
    knocked_up = np.zeros(N, dtype=bool)
    knocked_down = np.zeros(N, dtype=bool)

    for _ in range(M):
        dW = np.random.normal(0, sqrt_dt, N)

        S_up = np.maximum(0.0, S_up + r * S_up * dt + sigma * (S_up ** gamma) * dW)
        knocked_up |= (S_up >= B)

        S_down = np.maximum(0.0, S_down + r * S_down * dt + sigma * (S_down ** gamma) * dW)
        knocked_down |= (S_down >= B)

    S_up[knocked_up] = 0.0
    S_down[knocked_down] = 0.0

    v_up = np.exp(-r * T) * np.mean(payoff(S_up, k))
    v_down = np.exp(-r * T) * np.mean(payoff(S_down, k))

    return (v_up - v_down) / (2 * ds)


v_base = barrier_option_price(s0, k, r, T, B, sigma, gamma, N, M)
print(f"Baspris (sigma={sigma}, B={B}): {v_base:.4f}")

delta_base = calculate_delta(s0, k, r, T, B, sigma, gamma, 5000, M, ds)
print(f"Optionens delta (s0={s0}): {delta_base:.4f}")

v_sigma = barrier_option_price(s0, k, r, T, B, 0.2, gamma, N, M)
print(f"Andrad parameter (sigma=0.2): {v_sigma:.4f}")

v_barrier = barrier_option_price(s0, k, r, T, 25, sigma, gamma, N, M)
print(f"Andrad parameter (B=25): {v_barrier:.4f}")


s0_vals = np.linspace(11, 19.2, 14)
sigma_vals = np.linspace(0.1, 0.6, 14)
S0_grid, Sigma_grid = np.meshgrid(s0_vals, sigma_vals)

V_grid = np.zeros(S0_grid.shape)
D_grid = np.zeros(S0_grid.shape)

for i in range(len(sigma_vals)):
    for j in range(len(s0_vals)):
        s_val = S0_grid[i, j]
        sig_val = Sigma_grid[i, j]
        V_grid[i, j] = barrier_option_price(s_val, k, r, T, B, sig_val, gamma, 1000, 100)
        D_grid[i, j] = calculate_delta(s_val, k, r, T, B, sig_val, gamma, 1000, 100, ds)

fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(S0_grid, Sigma_grid, V_grid, cmap='viridis', edgecolor='none', alpha=0.9)
ax1.set_xlabel('Aktiepris (S0)')
ax1.set_ylabel('Volatilitet (sigma)')
ax1.set_zlabel('Optionspris (V)')
ax1.set_title(f'Optionspris V(S0, sigma) [B={B}, K={k}]')
cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=12, pad=0.1)
cbar1.set_label('Pris (V)')

ax2 = fig.add_subplot(122, projection='3d')
norm = TwoSlopeNorm(vmin=min(-0.1, D_grid.min()), vcenter=0, vmax=max(0.1, D_grid.max()))
surf2 = ax2.plot_surface(S0_grid, Sigma_grid, D_grid, cmap='coolwarm', norm=norm, edgecolor='none', alpha=0.9)

xx, yy = np.meshgrid(np.linspace(11, 19.2, 2), np.linspace(0.1, 0.6, 2))
ax2.plot_surface(xx, yy, np.zeros_like(xx), color='gray', alpha=0.25)

ax2.set_xlabel('Aktiepris (S0)')
ax2.set_ylabel('Volatilitet (sigma)')
ax2.set_zlabel('Delta (dV/dS0)')
ax2.set_title(f'Optionens Delta (dV/dS0) [B={B}, K={k}]')
cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=12, pad=0.1)
cbar2.set_label('Delta')

plt.tight_layout()
plt.savefig('barrier_price_and_delta.png', dpi=300)
plt.show()
