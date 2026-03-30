"""
Markowitz Efficient Frontier Portfolio Optimizer
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ── Constants ──────────────────────────────────────────────────────────────────
START_DATE = "2018-01-01"
END_DATE   = "2024-01-01"
TRADING_DAYS = 252
RISK_FREE_RATE = 0.06
N_SIMULATIONS = 20_000

# ── 1. User Inputs ─────────────────────────────────────────────────────────────
print("=" * 55)
print("   Markowitz Efficient Frontier Portfolio Optimizer")
print("=" * 55)

initial_investment = float(input("\nEnter initial investment amount ($): "))
n_assets = int(input("Enter number of assets: "))

tickers = []
for i in range(n_assets):
    t = input(f"  Enter ticker {i+1}: ").upper().strip()
    tickers.append(t)

# ── 2. Data Collection ─────────────────────────────────────────────────────────
print(f"\nDownloading data for {tickers} …")
raw = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

# Handle single-ticker edge case
if n_assets == 1:
    prices = raw[["Close"]].copy()
    prices.columns = tickers
else:
    prices = raw["Close"][tickers].copy()

prices.dropna(inplace=True)
print(f"Data ready: {len(prices)} trading days, {prices.shape[1]} assets.\n")

# ── 3. Portfolio Statistics ────────────────────────────────────────────────────
daily_returns = prices.pct_change().dropna()
ann_returns   = daily_returns.mean() * TRADING_DAYS          # vector
ann_cov       = daily_returns.cov()  * TRADING_DAYS          # matrix

# ── Helper functions ───────────────────────────────────────────────────────────

def portfolio_stats(weights):
    """Return (return, volatility, sharpe) for a weight vector."""
    weights = np.array(weights)
    ret  = float(weights @ ann_returns)
    vol  = float(np.sqrt(weights @ ann_cov.values @ weights))
    shrp = (ret - RISK_FREE_RATE) / vol
    return ret, vol, shrp


def sortino_ratio(weights):
    w = np.array(weights)
    port_daily = daily_returns.values @ w
    port_ret   = float(w @ ann_returns)
    downside   = port_daily[port_daily < 0]
    down_std   = np.std(downside) * np.sqrt(TRADING_DAYS) if len(downside) > 0 else 1e-9
    return (port_ret - RISK_FREE_RATE) / down_std


def historical_var(weights, confidence=0.95):
    w = np.array(weights)
    port_daily = daily_returns.values @ w
    return float(np.percentile(port_daily, (1 - confidence) * 100))


# ── 5. Optimization – Maximum Sharpe Ratio ─────────────────────────────────────
def neg_sharpe(weights):
    return -portfolio_stats(weights)[2]

constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
bounds      = [(0, 1)] * n_assets
x0          = np.full(n_assets, 1 / n_assets)

result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
opt_weights = result.x

# ── Equal-weight portfolio ─────────────────────────────────────────────────────
eq_weights = np.full(n_assets, 1 / n_assets)

# ── 6 & 7. Risk Metrics & Comparison ──────────────────────────────────────────
def full_metrics(weights, label):
    ret, vol, shrp = portfolio_stats(weights)
    sort  = sortino_ratio(weights)
    var95 = historical_var(weights)
    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"{'─'*40}")
    for ticker, w in zip(tickers, weights):
        print(f"  {ticker:8s}: {w*100:6.2f}%")
    print(f"  Expected Annual Return : {ret*100:.2f}%")
    print(f"  Annual Volatility      : {vol*100:.2f}%")
    print(f"  Sharpe Ratio           : {shrp:.4f}")
    print(f"  Sortino Ratio          : {sort:.4f}")
    print(f"  95% Historical VaR     : {var95*100:.2f}%  (daily)")
    return ret, vol, shrp, sort, var95

print("\n" + "=" * 55)
print("  PORTFOLIO COMPARISON")
opt_metrics = full_metrics(opt_weights, "Optimized Portfolio (Max Sharpe)")
eq_metrics  = full_metrics(eq_weights,  "Equal-Weight Portfolio")

# ── 8. Monte Carlo Simulation ─────────────────────────────────────────────────
print(f"\nRunning {N_SIMULATIONS:,} Monte Carlo simulations …")
mc_returns, mc_vols, mc_sharpes = [], [], []

for _ in range(N_SIMULATIONS):
    w = np.random.dirichlet(np.ones(n_assets))
    r, v, s = portfolio_stats(w)
    mc_returns.append(r)
    mc_vols.append(v)
    mc_sharpes.append(s)

mc_returns  = np.array(mc_returns)
mc_vols     = np.array(mc_vols)
mc_sharpes  = np.array(mc_sharpes)

# ── 9. Investment Growth ───────────────────────────────────────────────────────
opt_daily = (daily_returns.values @ opt_weights)
eq_daily  = (daily_returns.values @ eq_weights)

opt_growth = initial_investment * (1 + opt_daily).cumprod()
eq_growth  = initial_investment * (1 + eq_daily).cumprod()

dates = daily_returns.index

opt_final = opt_growth[-1]
eq_final  = eq_growth[-1]

print(f"\n{'='*55}")
print("  FINAL INVESTMENT VALUES")
print(f"  Initial Investment      : ${initial_investment:,.2f}")
print(f"  Optimized Portfolio     : ${opt_final:,.2f}  ({(opt_final/initial_investment-1)*100:.1f}%)")
print(f"  Equal-Weight Portfolio  : ${eq_final:,.2f}  ({(eq_final/initial_investment-1)*100:.1f}%)")
print("=" * 55)

# ── 10. Visualizations ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Markowitz Efficient Frontier – Portfolio Analysis", fontsize=14, fontweight="bold")

# — Graph 1: Efficient Frontier —
ax1 = axes[0]
sc = ax1.scatter(mc_vols, mc_returns, c=mc_sharpes, cmap="viridis", alpha=0.4, s=5)
plt.colorbar(sc, ax=ax1, label="Sharpe Ratio")

opt_r, opt_v, *_ = opt_metrics
eq_r,  eq_v,  *_ = eq_metrics

ax1.scatter(opt_v, opt_r, color="red",   s=200, zorder=5, label="Optimized (Max Sharpe)", marker="*")
ax1.scatter(eq_v,  eq_r,  color="orange", s=120, zorder=5, label="Equal Weight",           marker="D")

ax1.set_xlabel("Annual Volatility")
ax1.set_ylabel("Annual Return")
ax1.set_title("Efficient Frontier (Monte Carlo)")
ax1.legend()
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
ax1.grid(True, linestyle="--", alpha=0.4)

# — Graph 2: Investment Growth —
ax2 = axes[1]
ax2.plot(dates, opt_growth, color="red",    linewidth=1.8, label="Optimized Portfolio")
ax2.plot(dates, eq_growth,  color="orange", linewidth=1.8, label="Equal-Weight Portfolio")
ax2.axhline(initial_investment, color="grey", linestyle="--", linewidth=1, label="Initial Investment")

ax2.set_xlabel("Date")
ax2.set_ylabel("Portfolio Value ($)")
ax2.set_title("Investment Growth (2018 – 2024)")
ax2.legend()
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}"))
ax2.grid(True, linestyle="--", alpha=0.4)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/portfolio_analysis.png", dpi=150, bbox_inches="tight")
print("\nChart saved → portfolio_analysis.png")
plt.show()