# BrokerRLGymAnytrading


This project implements a Deep Q-Network (DQN) trading agent trained in multiple market environments using `gym-anytrading`. The agent learns to make Buy and Sell decisions based on historical stock and forex price data.

## Features

- Trains using DQN and PyTorch
- Multiple environments (Google stocks, Forex EURUSD)
- Periodic evaluation and graph export
- Model saving and resuming capabilities

## Train the agent

python main_training_loop.py

## Test the trained agent

python main_test_loop.py

## Final Results and Conclusions

After training the agent with various market segments and datasets for over 90,000 episodes, we observed that while the agent was able to develop basic trading behaviors, the overall performance remained limited — in some cases only performing buy actions, or achieving a final profit below break-even.

We believe that one of the main constraints lies in the limited customizability of the `gym-anytrading` environment. Features such as setting capital constraints, limiting the number of trades, enforcing cooldown periods, or applying transaction fees are not natively supported. These aspects are critical for simulating a more realistic trading scenario and guiding the agent towards more sophisticated strategies.

For these reasons, we have decided to close this project as a first exploratory step in reinforcement learning applied to trading. In future work, we aim to move to more advanced and flexible environments — such as [FinRL](https://github.com/AI4Finance-Foundation/FinRL) or custom Gym environments — that allow us to better model the market mechanics and exert finer control over the agent's interaction.

This project remains a solid baseline to build upon.
