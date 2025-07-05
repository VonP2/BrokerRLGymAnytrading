import gym 
import gym_anytrading
import yfinance as yf
import pandas as pd
from gym_anytrading.envs import StocksEnv
from gym_anytrading.datasets import STOCKS_GOOGL
import matplotlib.pyplot as plt
from graphs import plot_trading_summary
from graphs import plot_trading_summary_interactive
import pandas as pd

# Dataset
df= STOCKS_GOOGL.copy()

# Environment
env = StocksEnv(df=df, window_size=10, frame_bound=(10, 100))
actions_taken = []


obs = env.reset()
while True:
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)

    actions_taken.append(int(action))

    done = terminated or truncated
    if done:
       # Info Dictionary Cleaning
        clean_info = {
            "total_reward": float(info["total_reward"]),
            "total_profit": float(info["total_profit"]),
            "position": info["position"]
        }
        print(clean_info)

        graph_filename = f"graphs/prueba.html"

        plot_trading_summary_interactive(df, env, actions_taken,graph_filename)

        print(actions_taken)
        break


