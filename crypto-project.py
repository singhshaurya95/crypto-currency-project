import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime
import os


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"bitcoin_analysis_{timestamp}"
os.makedirs(output_dir, exist_ok=True)


def save_figure(fig, filename):
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close figure to free memory
    print(f"Saved: {path}")

# Get current date for dynamic end date
end_date = datetime.date.today().strftime('%Y-%m-%d')

# Download Bitcoin data for current year
BTC_USD = yf.download("BTC-USD", start='2025-01-01', end=end_date, interval='1d')

# Create price chart
fig1, ax = plt.subplots(figsize=(12, 6), dpi=100)
date_format = DateFormatter("%b-%d")
ax.xaxis.set_major_formatter(date_format)
ax.tick_params(axis='x', labelsize=9)
fig1.autofmt_xdate()
ax.plot(BTC_USD['Close'], lw=1.5, color='royalblue')
ax.set_ylabel('Price (USD)', fontsize=12)
ax.set_title(f'Bitcoin Price: Jan 1, 2025 - {end_date}', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
fig1.savefig("01_price_chart.png")

# Calculate moving averages
BTC_USD['SMA_9'] = BTC_USD['Close'].rolling(window=9, min_periods=1).mean()
BTC_USD['SMA_30'] = BTC_USD['Close'].rolling(window=30, min_periods=1).mean()

# Trading strategy implementation
trade_signals = pd.DataFrame(index=BTC_USD.index)
short_interval = 10
long_interval = 40

trade_signals['Short'] = BTC_USD['Close'].rolling(window=short_interval, min_periods=1).mean()
trade_signals['Long'] = BTC_USD['Close'].rolling(window=long_interval, min_periods=1).mean()
trade_signals['Signal'] = np.where(trade_signals['Short'] > trade_signals['Long'], 1.0, 0.0)
trade_signals['Position'] = trade_signals['Signal'].diff()

# Visualize trades
fig2, ax = plt.subplots(figsize=(12, 7), dpi=100)
ax.xaxis.set_major_formatter(date_format)
ax.tick_params(axis='x', labelsize=9)
fig2.autofmt_xdate()
fig2.savefig("02_trading_signals.png")

ax.plot(BTC_USD['Close'], lw=1.5, color='royalblue', label='Closing Price')
ax.plot(trade_signals['Short'], lw=1.5, alpha=0.8, color='orange', label=f'{short_interval}-Day SMA')
ax.plot(trade_signals['Long'], lw=1.5, alpha=0.8, color='purple', label=f'{long_interval}-Day SMA')

# Plot buy/sell signals
buy_signals = trade_signals.loc[trade_signals['Position'] == 1.0]
sell_signals = trade_signals.loc[trade_signals['Position'] == -1.0]

ax.plot(buy_signals.index, buy_signals.Short, '^', ms=10, color='green', label='Buy Signal')
ax.plot(sell_signals.index, sell_signals.Short, 'v', ms=10, color='red', label='Sell Signal')

ax.set_ylabel('Price (USD)', fontsize=12)
ax.set_title(f'Bitcoin Trading Signals: {short_interval}/{long_interval} SMA Crossover', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# Backtesting
initial_balance = 10000.0
backtest = pd.DataFrame(index=trade_signals.index)
backtest['BTC_Return'] = BTC_USD['Close'] / BTC_USD['Close'].shift(1)
backtest['Alg_Return'] = np.where(trade_signals.Signal == 1, backtest.BTC_Return, 1.0)
backtest['Balance'] = initial_balance * backtest.Alg_Return.cumprod()

# Portfolio comparison
fig3, ax = plt.subplots(figsize=(12, 6), dpi=100)
ax.xaxis.set_major_formatter(date_format)
ax.tick_params(axis='x', labelsize=9)
fig3.autofmt_xdate()
fig3.savefig("03_performance_comparison.png")

# Calculate buy-and-hold performance
buy_hold = initial_balance * backtest.BTC_Return.cumprod()

ax.plot(buy_hold, lw=1.5, alpha=0.8, color='blue', label='Buy and Hold')
ax.plot(backtest['Balance'], lw=1.5, alpha=0.8, color='darkorange', label='Crossover Strategy')

# Mark strategy trades
for date in buy_signals.index:
    if date in backtest.index:
        ax.axvline(x=date, color='green', alpha=0.3, linestyle='--')
for date in sell_signals.index:
    if date in backtest.index:
        ax.axvline(x=date, color='red', alpha=0.3, linestyle='--')

ax.set_ylabel('Portfolio Value (USD)', fontsize=12)
ax.set_title('Strategy Performance Comparison', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# Performance metrics
final_buy_hold = buy_hold[-1]
final_strategy = backtest['Balance'][-1]
strategy_trades = len(buy_signals) + len(sell_signals)

print(f"\n{' Strategy Performance Analysis ':=^50}")
print(f"Evaluation Period: Jan 1, 2025 - {end_date}")
print(f"Starting Capital: ${initial_balance:,.2f}")
print(f"\nBuy and Hold Final Value: ${final_buy_hold:,.2f}")
print(f"Crossover Strategy Final Value: ${final_strategy:,.2f}")
print(f"Strategy Outperformance: ${final_strategy - final_buy_hold:,.2f}")
print(f"\nTotal Trades Executed: {strategy_trades}")
print(f"Buy Signals: {len(buy_signals)}")
print(f"Sell Signals: {len(sell_signals)}")
print(f"\nStrategy Return: {(final_strategy/initial_balance-1)*100:.2f}%")
print(f"Buy/Hold Return: {(final_buy_hold/initial_balance-1)*100:.2f}%")


report = f"""
{' Strategy Performance Analysis ':=^50}
Evaluation Period: Jan 1, 2025 - {end_date}
Starting Capital: ${initial_balance:,.2f}

Buy and Hold Final Value: ${final_buy_hold:,.2f}
Crossover Strategy Final Value: ${final_strategy:,.2f}
Strategy Outperformance: ${final_strategy - final_buy_hold:,.2f}

Total Trades Executed: {strategy_trades}
Buy Signals: {len(buy_signals)}
Sell Signals: {len(sell_signals)}

Strategy Return: {(final_strategy/initial_balance-1)*100:.2f}%
Buy/Hold Return: {(final_buy_hold/initial_balance-1)*100:.2f}%
"""

# Save report to text file
report_path = os.path.join(output_dir, "performance_report.txt")
with open(report_path, 'w') as f:
    f.write(report)
print(f"Saved: {report_path}")

print(f"\nAnalysis complete! All outputs saved to: {output_dir}")
print(report)