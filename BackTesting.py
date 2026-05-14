!pip install pennylane --quiet

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import yfinance as yf
import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings("ignore")


STOCKS_COUNT = 256
QUBITS       = 8
CAPITAL      = 10000
RISK_LAMBDA  = 2.0
OPTIMIZATION_STEPS    = 1000


today         = datetime.today()
train_start   = today - timedelta(days=int(15 * 30.44))
train_end     = today - timedelta(days=int(3  * 30.44))
predict_start = train_end
predict_end   = today

print("=" * 60)
print(f"  Training Window  : {train_start.date()} → {train_end.date()}")
print(f"  Prediction Window: {predict_start.date()} → {predict_end.date()}")
print("=" * 60)

def fetch_tickers():
    url     = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp    = requests.get(url, headers=headers)
    df      = pd.read_html(StringIO(resp.text))[0]
    return [s.replace('.', '-') for s in df['Symbol'][:STOCKS_COUNT]]
symbols = fetch_tickers()

def download_prices(symbols, start, end):
    raw = yf.download(symbols, start=start.strftime('%Y-%m-%d'),
                      end=end.strftime('%Y-%m-%d'), progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close'] if 'Close' in raw.columns.levels[0] else raw.iloc[:, :len(symbols)]
    else:
        prices = raw.get('Close', raw)
    prices = prices.ffill().bfill().fillna(0)
    if prices.shape[1] < STOCKS_COUNT:
        pad = pd.DataFrame(0, index=prices.index,
                           columns=[f"CASH_{i}" for i in range(STOCKS_COUNT - prices.shape[1])])
        prices = pd.concat([prices, pad], axis=1)
    return prices.iloc[:, :STOCKS_COUNT]
train_prices   = download_prices(symbols, train_start,   train_end)
predict_prices = download_prices(symbols, predict_start, predict_end)

asset_names = train_prices.columns.tolist()

train_returns  = train_prices.pct_change().fillna(0)
avg_returns    = np.nan_to_num(train_returns.mean().values)
cov_matrix     = np.nan_to_num(train_returns.cov().values)
individual_risks   = np.sqrt(np.diag(cov_matrix))
individual_returns = avg_returns

assert STOCKS_COUNT == 2 ** QUBITS

dev = qml.device("default.qubit", wires=QUBITS)

@qml.qnode(dev)
def get_weights(params):
    qml.StronglyEntanglingLayers(weights=params, wires=range(QUBITS))
    return qml.probs(wires=range(QUBITS))

def objective(params):
    w = get_weights(params)
    portfolio_return = pnp.dot(w, pnp.asarray(avg_returns))
    portfolio_risk   = pnp.dot(w, pnp.dot(pnp.asarray(cov_matrix), w))
    return (RISK_LAMBDA * portfolio_risk) - portfolio_return

weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=6, n_wires=QUBITS)
theta         = pnp.random.random(weights_shape, requires_grad=True)
optimizer     = qml.AdamOptimizer(stepsize=0.01)


scores = []
for i in range(OPTIMIZATION_STEPS + 1):
    theta, score = optimizer.step_and_cost(objective, theta)
    scores.append(float(score))
    if i % 100 == 0:
        print(f"      Step {i:>3d} | Objective Score: {score:.6f}")
final_w = np.array(get_weights(theta))





predict_returns = predict_prices.pct_change().fillna(0)
final_w = np.array(get_weights(theta))
portfolio_daily_returns = predict_returns.values @ final_w          
portfolio_cum_returns   = np.cumprod(1 + portfolio_daily_returns)   
portfolio_values        = CAPITAL * portfolio_cum_returns           

predict_dates  = predict_prices.index.tolist()
n_predict_days = len(predict_dates)

equal_w              = np.ones(STOCKS_COUNT) / STOCKS_COUNT
bench_daily_returns  = predict_returns.values @ equal_w
bench_cum_returns    = np.cumprod(1 + bench_daily_returns)
bench_values         = CAPITAL * bench_cum_returns


final_portfolio_value = portfolio_values[-1]
final_bench_value     = bench_values[-1]
total_return_pct      = (final_portfolio_value - CAPITAL) / CAPITAL * 100
bench_return_pct      = (final_bench_value     - CAPITAL) / CAPITAL * 100


print(f"\n{'─'*55}")
print(f"  BACKTEST RESULTS  ({predict_start.date()} → {predict_end.date()})")
print(f"{'─'*55}")
print(f"  Starting Capital         : INR {CAPITAL:>15,.2f}")
print(f"  Final Portfolio Value    : INR {final_portfolio_value:>15,.2f}")
print(f"{'─'*55}")
print(f"  Total Return (Quantum)   : {total_return_pct:>+8.2f}%")
print(f"  Total Return (Benchmark) : {bench_return_pct:>+8.2f}%")
print(f"{'─'*55}")





output_dir = "/mnt/user-data/outputs/"
os.makedirs(output_dir, exist_ok=True)

plt.style.use('ggplot')
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Quantum Portfolio Optimizer — Backtest Analysis", fontsize=16, fontweight='bold', y=0.98)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])  
ax2 = fig.add_subplot(gs[0, 1])  
ax3 = fig.add_subplot(gs[1, 0])  



portfolio_volatality = np.sqrt(np.dot(final_w, np.dot(cov_matrix, final_w)))
portfolio_return = np.dot(final_w, avg_returns)

scatter = ax1.scatter(individual_risks, individual_returns,
                      alpha=0.35, color='royalblue', s=20, label='Individual Stocks')
ax1.scatter(portfolio_volatality, portfolio_return, color='red', s=200,
            edgecolors='black', zorder=6, label='Quantum Portfolio', marker='*')
ax1.set_title("Risk vs. Return (Training Window)")
ax1.set_xlabel("Daily Volatility (Risk)")
ax1.set_ylabel("Expected Daily Return")
ax1.legend(fontsize=8)



ax2.plot(scores, color='darkorange', linewidth=1.5)
ax2.set_title("Optimisation Convergence")
ax2.set_xlabel("Step")
ax2.set_ylabel("Objective Score")
ax2.axhline(y=min(scores), color='gray', linestyle='--', alpha=0.5, label=f"Best: {min(scores):.5f}")
ax2.legend(fontsize=8)



predict_date_range = range(n_predict_days)
ax3.plot(predict_date_range, portfolio_values, color='forestgreen',
         linewidth=2, label='Quantum Portfolio', zorder=3)
ax3.plot(predict_date_range, bench_values, color='steelblue',
         linewidth=1.5, linestyle='--', label='Equal-Weight Benchmark', zorder=2)
ax3.axhline(y=CAPITAL, color='black', linestyle=':', alpha=0.6, label='Starting Capital')
ax3.fill_between(predict_date_range, CAPITAL, portfolio_values,
                 where=(portfolio_values >= CAPITAL), alpha=0.15, color='green',  label='Gain')
ax3.fill_between(predict_date_range, CAPITAL, portfolio_values,
                 where=(portfolio_values <  CAPITAL), alpha=0.15, color='red',    label='Loss')


ax3.annotate(f"INR {final_portfolio_value:,.0f}\n({total_return_pct:+.2f}%)",
             xy=(n_predict_days - 1, final_portfolio_value),
             xytext=(-60, 12), textcoords='offset points',
             fontsize=8, color='forestgreen',
             arrowprops=dict(arrowstyle='->', color='forestgreen', lw=1))

ax3.set_title(f"Actual Portfolio Value — Prediction Window\n"
              f"({predict_start.date()} → {predict_end.date()})")
ax3.set_xlabel("Trading Days into Prediction Window")
ax3.set_ylabel("Portfolio Value (INR)")
ax3.legend(fontsize=8)
