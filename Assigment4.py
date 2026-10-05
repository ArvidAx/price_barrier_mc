import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

np.random.seed(42)

s0 = 14
k = 15
r = 0.05
T = 0.5
B = 20
N = 50000
M = 1000
sigma = 0.3
gamma = 1
ds = 0.1


def payoff(S, k):
    return np.maximum(S - k, 0.0)


def barrier_option_price(s0, k, r, T, B, sigma, gamma, N, M):
    dt = T / M
    sqrt_dt = np.sqrt(dt)

    S = np.full(N, s0)
    knocked_out = S >= B

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
    knocked_up = S_up >= B
    knocked_down = S_down >= B

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


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(s0, k, r, T, sigma):
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return s0 * norm_cdf(d1) - k * math.exp(-r * T) * norm_cdf(d2), norm_cdf(d1)


def simulate_paths(s0, r, T, sigma, gamma, n_paths, M):
    dt = T / M
    paths = np.zeros((n_paths, M + 1))
    paths[:, 0] = s0
    for m in range(M):
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        S = paths[:, m]
        paths[:, m + 1] = np.maximum(0.0, S + r * S * dt + sigma * (S ** gamma) * dW)
    return paths


v_base = barrier_option_price(s0, k, r, T, B, sigma, gamma, N, M)
delta_base = calculate_delta(s0, k, r, T, B, sigma, gamma, N, M, ds)
v_vanilla, delta_vanilla = bs_call(s0, k, r, T, sigma)
print(f"Barrier (sigma={sigma}, B={B}): pris {v_base:.4f}, delta {delta_base:.4f}")
print(f"Vanilla (Black-Scholes):        pris {v_vanilla:.4f}, delta {delta_vanilla:.4f}")


# Barrier vs vanilla: pris mot S0 och mot sigma
s0_line = np.linspace(10, 20, 21)
v_s0_bar = [barrier_option_price(s, k, r, T, B, sigma, gamma, 20000, 500) for s in s0_line]
v_s0_van = [bs_call(s, k, r, T, sigma)[0] for s in s0_line]

sigma_line = np.linspace(0.05, 0.6, 12)
v_sig_bar = [barrier_option_price(s0, k, r, T, B, sig, gamma, 20000, 500) for sig in sigma_line]
v_sig_van = [bs_call(s0, k, r, T, sig)[0] for sig in sigma_line]
for sig, vb, vv in zip(sigma_line, v_sig_bar, v_sig_van):
    print(f"sigma={sig:.2f}: barrier {vb:.4f}, vanilla {vv:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(s0_line, v_s0_van, '--', color='gray', lw=2, label='Vanilla call')
ax1.plot(s0_line, v_s0_bar, 'o-', color='tab:blue', lw=2, label='Up-and-out call')
ax1.axvline(B, color='black', lw=1, ls=':')
ax1.set_xlabel('Stock price S0')
ax1.set_ylabel('Option price')
ax1.set_title(f'Price vs stock price (sigma = {sigma}, B = {B})')
ax1.legend()
ax2.plot(sigma_line, v_sig_van, '--', color='gray', lw=2, label='Vanilla call')
ax2.plot(sigma_line, v_sig_bar, 'o-', color='tab:blue', lw=2, label='Up-and-out call')
ax2.set_xlabel('Volatility sigma')
ax2.set_ylabel('Option price')
ax2.set_title(f'Price vs volatility (S0 = {s0}, B = {B})')
ax2.legend()
plt.tight_layout()
plt.savefig('barrier_vs_vanilla.png', dpi=200)


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
ax1.set_xlabel('Stock price (S0)')
ax1.set_ylabel('Volatility (sigma)')
ax1.set_zlabel('Option price (V)')
ax1.set_title(f'Option price V(S0, sigma) [B={B}, K={k}]')
cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=12, pad=0.1)
cbar1.set_label('Price (V)')

ax2 = fig.add_subplot(122, projection='3d')
norm = TwoSlopeNorm(vmin=min(-0.1, D_grid.min()), vcenter=0, vmax=max(0.1, D_grid.max()))
surf2 = ax2.plot_surface(S0_grid, Sigma_grid, D_grid, cmap='coolwarm', norm=norm, edgecolor='none', alpha=0.9)

xx, yy = np.meshgrid(np.linspace(11, 19.2, 2), np.linspace(0.1, 0.6, 2))
ax2.plot_surface(xx, yy, np.zeros_like(xx), color='gray', alpha=0.25)

ax2.set_xlabel('Stock price (S0)')
ax2.set_ylabel('Volatility (sigma)')
ax2.set_zlabel('Delta (dV/dS0)')
ax2.set_title(f'Option delta (dV/dS0) [B={B}, K={k}]')
cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=12, pad=0.1)
cbar2.set_label('Delta')

plt.tight_layout()
plt.savefig('barrier_price_and_delta.png', dpi=300)
plt.show()


# Sanity checks: B mycket stor ger vanilla, s0 = B ger 0
print(f"Sanity check B=1000: {barrier_option_price(s0, k, r, T, 1000, sigma, gamma, N, M):.4f} (vanilla {v_vanilla:.4f})")
print(f"Sanity check s0=B=20: {barrier_option_price(20, k, r, T, B, sigma, gamma, N, M):.4f}")


# Simulerade banor (eget seed sa att figuren visar bade utslagna och overlevande banor)
np.random.seed(13)
paths =simulate_paths(s0, r, T, 0.3, gamma, 8, M)
t_grid = np.linspace(0, T, M + 1)
fig, ax = plt.subplots(figsize=(10, 5.5))
for p in paths:
    hit = np.argmax(p >= B) if np.any(p >= B) else None
    if hit is None:
        ax.plot(t_grid, p, color='tab:green', lw=1.5)
    else:
        ax.plot(t_grid[:hit + 1], p[:hit + 1], color='tab:red', lw=1.5)
        ax.plot(t_grid[hit:], p[hit:], color='tab:red', lw=1.0, alpha=0.25, ls='--')
        ax.plot(t_grid[hit], p[hit], 'x', color='tab:red', ms=10, mew=2)
ax.axhline(B, color='black', lw=2, label=f'Barrier B = {B}')
ax.axhline(k, color='gray', lw=1.5, ls=':', label=f'Strike K = {k}')
ax.plot([], [], color='tab:green', label='Survives (pays call payoff)')
ax.plot([], [], color='tab:red', label='Knocked out (pays 0)')
ax.set_xlabel('Time t (years)')
ax.set_ylabel('Stock price S(t)')
ax.set_title('Simulated paths, up-and-out call (sigma = 0.3)')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('barrier_sample_paths.png', dpi=200)


# Konvergens i antal banor N (sigma = 0.3, dar barriaren spelar roll)
sigma_conv = 0.3
v_bs_conv, delta_bs_conv = bs_call(s0, k, r, T, sigma_conv)
N_vals = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
v_conv = [barrier_option_price(s0, k, r, T, B, sigma_conv, gamma, n, M) for n in N_vals]
d_conv = [calculate_delta(s0, k, r, T, B, sigma_conv, gamma, n, M, ds) for n in N_vals]
for n, v, d in zip(N_vals, v_conv, d_conv):
    print(f"N={n:6d}: pris {v:.4f}, delta {d:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.semilogx(N_vals, v_conv, 'o-', color='tab:blue', label='Barrier option (MC)')
ax1.axhline(v_bs_conv, color='gray', ls='--', label=f'Vanilla call (BS) = {v_bs_conv:.4f}')
ax1.set_xlabel('Number of paths N')
ax1.set_ylabel('Option price')
ax1.set_title(f'Price vs number of paths (sigma = {sigma_conv})')
ax1.legend()
ax2.semilogx(N_vals, d_conv, 'o-', color='tab:orange', label='Barrier option delta (MC)')
ax2.axhline(delta_bs_conv, color='gray', ls='--', label=f'Vanilla call delta (BS) = {delta_bs_conv:.4f}')
ax2.set_xlabel('Number of paths N')
ax2.set_ylabel('Delta')
ax2.set_title(f'Delta vs number of paths (sigma = {sigma_conv})')
ax2.legend()
plt.tight_layout()
plt.savefig('barrier_convergence.png', dpi=200)
plt.show()
