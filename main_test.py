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
from dqn_action_trading_agent import DQNAgent
import os


class MainTest():
    def __init__(self):
        self.df = STOCKS_GOOGL.copy()
        self.env = StocksEnv(df=self.df, window_size=10, frame_bound=(210, 300))
        self.actions_taken = []
        self.agent_name = "DQNBroker"


        state_size = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        self.agent = DQNAgent(state_size, self.env.action_space.n)

        if os.path.exists(self.agent_name):
            print(f"Loading model from '{self.agent_name}'...")
            self.agent.load(self.agent_name)
        else:
            print("Training new model...")


    def run(self):

        self.actions_taken = []
        state, _ = self.env.reset()
        self.done = False


        while not self.done:

                action = self.agent.act(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                state = next_state

                self.actions_taken.append(int(action))
                self.done = terminated or truncated

                if self.done:
                    final_info = info

        clean_info = {
                "total_reward": float(final_info["total_reward"]),
                "total_profit": float(final_info["total_profit"]),
                "position": final_info["position"]
        }

        graph_filename = f"graphs/DQNBrokerGraph_test.html"
        plot_trading_summary_interactive(self.df, self.env, self.actions_taken, graph_filename)
        print(f"Graph saved: {graph_filename}")
        print(clean_info)
                        
    
if __name__ == "__main__":
    main_loop = MainTest()
    main_loop.run()

