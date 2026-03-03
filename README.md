# Pacman Python Pygame

A Pacman game built with Python and Pygame, featuring a Deep Q-Network (DQN) reinforcement learning agent.

## Requirements

```bash
pip install torch numpy pygame tcod
```

## How to Use

### 1. Run training
```bash
python pacman_rl.py --mode train
```

### 2. Watch the trained agent play
```bash
python pacman_rl.py --mode play
```

### 3. Continue training from a checkpoint
```bash
python pacman_rl.py --mode train --model pacman_dqn.pth
```
