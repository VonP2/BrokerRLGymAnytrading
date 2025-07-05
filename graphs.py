import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_trading_summary(df, env, actions):
    plt.ion()

    prices = df['Close'].values[env.frame_bound[0]:env.frame_bound[1]]
    profit_history = env.history['total_profit']

    buy_signals = []
    sell_signals = []

    for i, action in enumerate(actions):
        if action == 1:
            buy_signals.append(i)
        elif action == 0:
            sell_signals.append(i)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    ax1.plot(profit_history, label='Profit (%)', color='blue')
    ax1.axhline(y=1, color='red', linestyle='--', label='Break-even')
    ax1.set_title("📈 Portfolio Evolution")
    ax1.set_ylabel("Profit")
    ax1.legend()
    ax1.grid()

    ax2.plot(prices, label='Stock Price', color='black')
    if buy_signals:
        ax2.scatter(buy_signals, prices[buy_signals], marker='^', color='green', label='Buy', alpha=0.6)
    if sell_signals:
        ax2.scatter(sell_signals, prices[sell_signals], marker='v', color='red', label='Sell', alpha=0.6)
    ax2.set_title("💸 Trading Actions")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Price")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()
    plt.savefig('graficaAgente.png')
    input("Press ENTER to close the graphic...")
    plt.close()


def plot_trading_summary_interactive(df, env, actions, save_path):
    prices = df['Close'].values[env.frame_bound[0]:env.frame_bound[1]]
    profit_history = env.history['total_profit']

    buy_signals = []
    sell_signals = []

    for i, action in enumerate(actions):
        if action == 1:
            buy_signals.append(i)
        elif action == 0:
            sell_signals.append(i)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("📈 Portfolio Profit", "💸 Trading Actions")
    )

    fig.add_trace(go.Scatter(
        y=profit_history,
        mode='lines',
        name='Profit (%)',
        line=dict(color='blue')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        y=[1]*len(profit_history),
        mode='lines',
        name='Break-even',
        line=dict(color='red', dash='dash')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        y=prices,
        mode='lines',
        name='Stock Price',
        line=dict(color='black')
    ), row=2, col=1)

    if buy_signals:
        fig.add_trace(go.Scatter(
            x=buy_signals,
            y=[prices[i] for i in buy_signals],
            mode='markers',
            name='Buy',
            marker=dict(symbol='triangle-up', color='green', size=10)
        ), row=2, col=1)

    if sell_signals:
        fig.add_trace(go.Scatter(
            x=sell_signals,
            y=[prices[i] for i in sell_signals],
            mode='markers',
            name='Sell',
            marker=dict(symbol='triangle-down', color='red', size=10)
        ), row=2, col=1)

    fig.update_layout(
        height=700,
        title_text="Trading Agent Summary",
        template="plotly_white",
        showlegend=True
    )

    fig.update_yaxes(title_text="Profit", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    fig.write_html(save_path)
    print(f"Graph saved in: {save_path}")