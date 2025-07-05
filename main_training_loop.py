import gym 
import gym_anytrading
import yfinance as yf
import pandas as pd
from gym_anytrading.envs import StocksEnv
from gym_anytrading.datasets import STOCKS_GOOGL, FOREX_EURUSD_1H_ASK
import yfinance as yf
import matplotlib.pyplot as plt
from graphs import plot_trading_summary
from graphs import plot_trading_summary_interactive
import pandas as pd
from dqn_action_trading_agent import DQNAgent
import os
import random


class MainLoop():
    def __init__(self):
        self.envs = [
            ("GOOGL_10_110", StocksEnv(df=STOCKS_GOOGL.copy(), window_size=10, frame_bound=(10, 110))),
            ("GOOGL_510_700", StocksEnv(df=STOCKS_GOOGL.copy(), window_size=10, frame_bound=(510, 700))),
            ("FOREX_10_100", StocksEnv(df=FOREX_EURUSD_1H_ASK.copy(), window_size=10, frame_bound=(10, 100))),
            ("FOREX_110_210", StocksEnv(df=FOREX_EURUSD_1H_ASK.copy(), window_size=10, frame_bound=(110, 210))),
        ]
        self.actions_taken = []
        self.done = False
        self.agent_name = "DQNBroker"
        self.save_path = "DQNBrokerGraph.html"

        state_size = self.envs[0][1].observation_space.shape[0] * self.envs[0][1].observation_space.shape[1]
        self.agent = DQNAgent(state_size, self.envs[0][1].action_space.n)

        if os.path.exists(self.agent_name):
            print(f"Loading model from '{self.agent_name}'...")
            self.agent.load(self.agent_name)
        else:
            print("Training new model...")


    def run(self):
        while self.agent.keep_training():
            env_name,self.env = random.choice(self.envs)
            self.df = self.env.df
            state, _ = self.env.reset()
            self.done = False
            self.actions_taken = []

            while not self.done:

                action = self.agent.act(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                self.agent.remember(state, action, reward, next_state, terminated or truncated)
                
                if len(self.agent.memory)> self.agent.batch_size:
                    self.agent.replay(self.agent.batch_size)

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

            self.agent.epsilon_greedy()

            self.agent.episodes_done += 1

            print(self.agent.episodes_done)

            if self.agent.episodes_done % 10 == 0:
                self.agent.update_target_network()

            if self.agent.episodes_done % 100 == 0:
                print(self.agent.epsilon)
                self.agent.save(self.agent_name, self.agent.epsilon)

            if self.agent.episodes_done % 1000 == 0:
                graph_filename = f"graphs/DQNBrokerGraph_{env_name}_{self.agent.episodes_done}.html"
                plot_trading_summary_interactive(self.df, self.env, self.actions_taken, graph_filename)
                print(f"Graph saced: {graph_filename}")
                        
    
if __name__ == "__main__":
    main_loop = MainLoop()
    main_loop.run()

