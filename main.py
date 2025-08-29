import yfinance as yf
import pandas as pd
import numpy as np
import os
import argparse
from agent import Agent

# data parameters
symbol = "SAAB-B.ST"
interval = "1d"
period = "5y"

# hyperparameters
episodes = 15000
lr = 1e-4
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32

def fetch_data(symbol, interval, period):
    """Fetch stock data from Yahoo Finance"""
    stock_data = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
    os.makedirs("data", exist_ok=True)
    stock_data.to_csv(os.path.join("data", f"{symbol}_{period}_{interval}_data.csv"))
    return stock_data

def load_data(data_path):
    """Load and preprocess stock data"""
    data = pd.read_csv(data_path)
    
    if 'Price' in data.columns:
        data = data.iloc[2:]
        data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data = data.dropna()
    return data

def split_data(data, train_ratio=0.7):
    """Split data into training and testing sets"""
    split_index = int(len(data) * train_ratio)
    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()
    return train_data, test_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Trader")
    parser.add_argument("--mode", choices=["train", "eval"], required=True, help="Mode to run the agent")
    args = parser.parse_args()

    # if data does not exist, fetch it
    data_path = os.path.join("data", f"{symbol}_{period}_{interval}_data.csv")
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Fetching data...")
        fetch_data(symbol, interval, period)
    
    # Load and split data
    data = load_data(data_path)
    train_data, test_data = split_data(data, train_ratio=0.7)
    
    print(f"Total data points: {len(data)}")
    print(f"Training data points: {len(train_data)}")
    print(f"Testing data points: {len(test_data)}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Training range: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Testing range: {test_data.index[0]} to {test_data.index[-1]}")

    agent = Agent(
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )

    if args.mode == "train":
        print(f"\n🚀 Starting training on {symbol} data...")
        agent.train(train_data, episodes=episodes)
        print("Training completed!")
    elif args.mode == "eval":
        print(f"\n📊 Starting evaluation on {symbol} test data...")
        agent.evaluate(test_data, ticker=symbol)
        print("Evaluation completed!")