import gymnasium as gym
import numpy as np
import pandas as pd

class Env(gym.Env):
    def __init__(self, data, initial_balance=1000, slippage=0.0005, trading_fee=0.001, lookback_window=10):
        """
        environment for trading agent
        
        args:
        data: pandas DataFrame containing historical market data
        initial_balance: initial balance for the trading agent
        slippage: slippage factor for trades (0.0005 for 0.05% slippage)
        trading_fee: trading fee factor (0.001 for 0.1% trading fee)
        lookback_window: number of previous time steps to consider for observations
        """
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.slippage = slippage
        self.trading_fee = trading_fee
        self.lookback_window = lookback_window
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation space: price features + position info
        obs_dim = len(data.columns) * lookback_window + 8  # price data + account info
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Trading state
        self.reset()

    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.done = False
        
        return self._get_observation()

    def _get_observation(self):
        """Get current state observation"""
        # Price data for lookback window (normalized)
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        price_data = self.data.iloc[start_idx:end_idx]
        
        # Normalize price data
        if len(price_data) > 0:
            price_features = []
            for col in self.data.columns:
                if col in ['Open', 'High', 'Low', 'Close']:
                    # Percentage change normalization
                    values = price_data[col].values
                    if len(values) > 1:
                        pct_changes = np.diff(values) / values[:-1]
                        # Pad with zeros if needed
                        while len(pct_changes) < self.lookback_window - 1:
                            pct_changes = np.concatenate([[0], pct_changes])
                        price_features.extend(pct_changes[-self.lookback_window + 1:])
                    else:
                        price_features.extend([0] * (self.lookback_window - 1))
                elif col == 'Volume':
                    # Log normalization for volume
                    values = price_data[col].values
                    if len(values) > 0:
                        log_volumes = np.log(values + 1)
                        # Normalize by rolling mean
                        if len(log_volumes) > 1:
                            normalized = (log_volumes - np.mean(log_volumes)) / (np.std(log_volumes) + 1e-8)
                        else:
                            normalized = [0]
                        # Pad with zeros if needed
                        while len(normalized) < self.lookback_window:
                            normalized = np.concatenate([[0], normalized])
                        price_features.extend(normalized[-self.lookback_window:])
                    else:
                        price_features.extend([0] * self.lookback_window)
        else:
            price_features = [0] * (len(self.data.columns) * self.lookback_window)
        
        # Account information (normalized)
        current_price = self.data.iloc[self.current_step]['Close']
        account_features = [
            self.balance / self.initial_balance,  # Normalized balance
            self.shares_held * current_price / self.initial_balance,  # Normalized position value
            self.net_worth / self.initial_balance,  # Normalized net worth
            (self.net_worth - self.initial_balance) / self.initial_balance,  # Return
            self.shares_held / 100,  # Normalized shares held
            float(self.shares_held > 0),  # Position indicator
            self.current_step / len(self.data),  # Time progress
            (current_price - self.data['Close'].mean()) / self.data['Close'].std()  # Price z-score
        ]
        
        observation = np.array(price_features + account_features, dtype=np.float32)
        return observation

    def step(self, action):
        """Take a step in the environment"""
        if self.done:
            return self._get_observation(), 0, True, {}
        
        current_price = self.data.iloc[self.current_step]['Close']
        reward = 0
        
        # Execute action
        if action == 1:  # Buy
            reward = self._buy(current_price)
        elif action == 2:  # Sell
            reward = self._sell(current_price)
        # action == 0 is hold, no trade
        
        # Update net worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # Calculate reward based on net worth change
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        
        # Step reward (change in net worth)
        prev_net_worth = self.max_net_worth if self.current_step == self.lookback_window else self._get_prev_net_worth()
        reward += (self.net_worth - prev_net_worth) / self.initial_balance
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        self.done = self.current_step >= len(self.data) - 1
        
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'net_worth': self.net_worth,
            'current_price': current_price
        }
        
        return self._get_observation(), reward, self.done, info

    def _get_prev_net_worth(self):
        """Calculate previous net worth"""
        if self.current_step > self.lookback_window:
            prev_price = self.data.iloc[self.current_step - 1]['Close']
            return self.balance + (self.shares_held * prev_price)
        return self.initial_balance

    def _buy(self, current_price):
        """Execute buy action"""
        # Calculate maximum shares we can buy
        effective_price = current_price * (1 + self.slippage + self.trading_fee)
        max_shares = int(self.balance // effective_price)
        
        if max_shares > 0:
            shares_to_buy = max_shares
            cost = shares_to_buy * effective_price
            
            self.balance -= cost
            self.shares_held += shares_to_buy
            
            # Small reward for successful buy when trend is positive
            return 0.01
        
        return 0  # No reward if can't buy

    def _sell(self, current_price):
        """Execute sell action"""
        if self.shares_held > 0:
            effective_price = current_price * (1 - self.slippage - self.trading_fee)
            revenue = self.shares_held * effective_price
            
            self.balance += revenue
            self.total_shares_sold += self.shares_held
            self.total_sales_value += revenue
            self.shares_held = 0
            
            # Small reward for successful sell
            return 0.01
        
        return 0  # No reward if no shares to sell

    def render(self, mode='human'):
        """Render the environment"""
        current_price = self.data.iloc[self.current_step]['Close'] if self.current_step < len(self.data) else 0
        profit = self.net_worth - self.initial_balance
        
        print(f'Step: {self.current_step}')
        print(f'Balance: ${self.balance:.2f}')
        print(f'Shares held: {self.shares_held}')
        print(f'Current price: ${current_price:.2f}')
        print(f'Net worth: ${self.net_worth:.2f}')
        print(f'Profit: ${profit:.2f} ({profit/self.initial_balance*100:.2f}%)')
        print('---')