from model import DQN
import copy
import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from environment import Env
import os
import matplotlib.pyplot as plt

class Agent():
    def __init__(
            self, 
            state_dim = None,  # Will be set based on environment
            action_dim = 3,  # buy, sell, hold
            lr = 1e-4,
            gamma = 0.99,
            epsilon = 1.0,
            epsilon_min = 0.01,
            epsilon_decay = 0.995,
            batch_size = 32
            ):
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        
        # the fun stuff, mess around with the agent's parameters
        self.learning_rate = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Environment and networks will be initialized when train/eval is called
        self.env = None
        self.q_net = None
        self.target_net = None
        self.optimizer = None

        # use cuda if available
        self.device = self.choose_device()
        print(f"Using device: {self.device}")
    
    def _initialize_networks(self, state_dim):
        """Initialize networks once we know the state dimension"""
        if self.q_net is None:
            self.state_dim = state_dim
            self.q_net = DQN(state_dim, 128, self.action_dim)
            self.target_net = copy.deepcopy(self.q_net)
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate) # testa med Adam
            self.q_net.to(self.device)
            self.target_net.to(self.device)
    
    def choose_device(self):
        """choose the device for computation, CUDA if available, elif metal/mps else CPU"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def act(self, state):
        """choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_net(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """train the agent using replay memory"""
        if len(self.memory) < self.batch_size:
            return 0

        # sample a batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(np.array(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_tensor = torch.FloatTensor(np.array(dones)).to(self.device)

        # compute target Q-values
        with torch.no_grad():
            target_q_values = self.target_net(next_states_tensor)
            target_q_values = rewards_tensor + self.gamma * target_q_values.max(dim=1)[0] * (1 - dones_tensor)

        # compute current Q-values
        current_q_values = self.q_net(states_tensor)
        current_q_values = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # compute loss and optimize
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def decay_epsilon(self):
        """decay epsilon after each episode"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """update target network with current Q-network weights"""
        if self.target_net is not None:
            self.target_net.load_state_dict(self.q_net.state_dict())
    
    def train(self, train_data, episodes=1000):
        """Train the agent on the training data"""
        print(f"Training agent for {episodes} episodes...")
        
        # Create training environment
        self.env = Env(train_data)
        
        # Initialize networks with correct state dimension
        sample_state = self.env.reset()
        self._initialize_networks(len(sample_state))
        
        # Training metrics
        episode_rewards = []
        episode_losses = []
        net_worths = []
        
        # Track best performance for saving best model
        best_net_worth = 0
        best_episode = 0
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            total_loss = 0
            steps = 0
            
            while not self.env.done:
                # Choose action
                action = self.act(state)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                self.remember(state, action, reward, next_state, done)
                
                # Train network
                loss = self.replay()
                if loss:
                    total_loss += loss
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # Update target network every 10 episodes
            if episode % 10 == 0:
                self.update_target_network()
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Record metrics
            episode_rewards.append(total_reward)
            episode_losses.append(total_loss / max(steps, 1))
            net_worths.append(self.env.net_worth)
            
            # Save best model
            if self.env.net_worth > best_net_worth:
                best_net_worth = self.env.net_worth
                best_episode = episode
                self.save("dqn_trader_best.pth")
                print(f"💾 New best model saved! Episode {episode}: Net Worth ${self.env.net_worth:.2f} (Profit: ${self.env.net_worth - self.env.initial_balance:.2f})")
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                avg_loss = np.mean(episode_losses[-10:]) if len(episode_losses) >= 10 else np.mean(episode_losses)
                print(f"Episode {episode}/{episodes}")
                print(f"  Avg Reward (last 10): {avg_reward:.4f}")
                print(f"  Avg Loss (last 10): {avg_loss:.4f}")
                print(f"  Epsilon: {self.epsilon:.4f}")
                print(f"  Net Worth: ${self.env.net_worth:.2f}")
                print(f"  Profit: ${self.env.net_worth - self.env.initial_balance:.2f}")
                print(f"  Best Net Worth: ${best_net_worth:.2f} (Episode {best_episode})")
                print("-" * 50)
        
        # Save final model
        self.save("dqn_trader_final.pth")
        print(f"Final model saved to models/dqn_trader_final.pth")
        
        # Summary of best performance
        print(f"\n🏆 TRAINING SUMMARY:")
        print(f"Best performance: ${best_net_worth:.2f} at episode {best_episode}")
        print(f"Best profit: ${best_net_worth - self.env.initial_balance:.2f}")
        print(f"Best return: {((best_net_worth - self.env.initial_balance) / self.env.initial_balance) * 100:.2f}%")
        print(f"Best model saved as: models/dqn_trader_best.pth")
        
        # Plot training metrics
        self._plot_training_results(episode_rewards, episode_losses, net_worths)
        
        return episode_rewards, episode_losses, net_worths
    
    def evaluate(self, test_data, ticker="STOCK"):
        """Evaluate the agent on test data"""
        print(f"Evaluating agent on {ticker} test data...")
        
        # Try to load best model first, then fall back to other models
        model_paths = [
            "models/dqn_trader_best.pth",
            "models/dqn_trader_final.pth", 
            "models/dqn_trader.pth"
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                # Create evaluation environment first to get state dimension
                eval_env = Env(test_data)
                sample_state = eval_env.reset()
                self._initialize_networks(len(sample_state))
                
                self.load(model_path)
                print(f"Loaded model from {model_path}")
                model_loaded = True
                break
        
        if not model_loaded:
            print("No trained model found. Please train the agent first.")
            return
        
        # Set epsilon to 0 for evaluation (no exploration)
        original_epsilon = self.epsilon
        self.epsilon = 0.0
        
        # Create evaluation environment
        self.env = Env(test_data)
        state = self.env.reset()
        
        # Evaluation metrics
        actions_taken = []
        rewards = []
        net_worths = []
        prices = []
        
        step = 0
        while not self.env.done:
            # Choose action (greedy)
            action = self.act(state)
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            
            # Record metrics
            actions_taken.append(action)
            rewards.append(reward)
            net_worths.append(info['net_worth'])
            prices.append(info['current_price'])
            
            state = next_state
            step += 1
        
        # Restore original epsilon
        self.epsilon = original_epsilon
        
        # Calculate final results
        initial_balance = self.env.initial_balance
        final_net_worth = self.env.net_worth
        total_profit = final_net_worth - initial_balance
        total_return = (total_profit / initial_balance) * 100
        
        # Buy and hold comparison
        initial_price = test_data.iloc[0]['Close']
        final_price = test_data.iloc[-1]['Close']
        buy_hold_return = ((final_price - initial_price) / initial_price) * 100
        
        print(f"\n{'='*50}")
        print(f"EVALUATION RESULTS FOR {ticker}")
        print(f"{'='*50}")
        print(f"Initial Balance: ${initial_balance:.2f}")
        print(f"Final Net Worth: ${final_net_worth:.2f}")
        print(f"Total Profit: ${total_profit:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
        print(f"Outperformance: {total_return - buy_hold_return:.2f}%")
        print(f"Total Steps: {step}")
        print(f"Actions taken - Hold: {actions_taken.count(0)}, Buy: {actions_taken.count(1)}, Sell: {actions_taken.count(2)}")
        print(f"{'='*50}")
        
        # Plot evaluation results
        self._plot_evaluation_results(net_worths, prices, actions_taken, ticker)
        
        return {
            'initial_balance': initial_balance,
            'final_net_worth': final_net_worth,
            'total_profit': total_profit,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'actions': actions_taken,
            'net_worths': net_worths,
            'prices': prices
        }
    
    def _plot_training_results(self, rewards, losses, net_worths):
        """Plot training results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        ax1.plot(rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Episode losses
        ax2.plot(losses)
        ax2.set_title('Episode Losses')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        
        # Net worth progression
        ax3.plot(net_worths)
        ax3.set_title('Net Worth Over Episodes')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Net Worth ($)')
        ax3.grid(True)
        
        # Moving averages
        if len(rewards) > 100:
            window = 100
            moving_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax4.plot(moving_rewards)
            ax4.set_title(f'Moving Average Rewards (window={window})')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Avg Reward')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_evaluation_results(self, net_worths, prices, actions, ticker):
        """Plot evaluation results"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        steps = range(len(net_worths))
        
        # Net worth over time
        ax1.plot(steps, net_worths, label='Agent Net Worth', linewidth=2)
        ax1.set_title(f'{ticker} - Agent Performance')
        ax1.set_ylabel('Net Worth ($)')
        ax1.grid(True)
        ax1.legend()
        
        # Price over time
        ax2.plot(steps, prices, label=f'{ticker} Price', color='orange', linewidth=2)
        ax2.set_title(f'{ticker} - Price Movement')
        ax2.set_ylabel('Price ($)')
        ax2.grid(True)
        ax2.legend()
        
        # Actions taken
        buy_points = [i for i, action in enumerate(actions) if action == 1]
        sell_points = [i for i, action in enumerate(actions) if action == 2]
        
        ax3.scatter(buy_points, [prices[i] for i in buy_points], color='green', label='Buy', alpha=0.7, s=30)
        ax3.scatter(sell_points, [prices[i] for i in sell_points], color='red', label='Sell', alpha=0.7, s=30)
        ax3.plot(steps, prices, color='orange', alpha=0.3, linewidth=1)
        ax3.set_title(f'{ticker} - Trading Actions')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Price ($)')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(f'{ticker}_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save(self, filename):
        """save the agent's model to a file"""
        if self.q_net is not None:
            torch.save(self.q_net.state_dict(), f"models/{filename}")

    def load(self, filepath):
        """load the agent's model from a filepath"""
        if self.q_net is not None:
            self.q_net.load_state_dict(torch.load(filepath, map_location=self.device))
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.q_net.to(self.device)
            self.target_net.to(self.device)