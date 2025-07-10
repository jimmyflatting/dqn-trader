# dqn trader

## usage

```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
# for training
python main.py --mode train

# for eval (takes current best model)
python main.py --mode eval
```

## hyperparams in `main.py`

```bash
# data parameters
symbol = "AAPL"
interval = "1d"
period = "1y"

# hyperparameters
episodes = 1000
lr = 1e-4
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
```
