# SuperTuxKart RL Bot

A robust, multi-processed Reinforcement Learning (RL) bot that learns to navigate the SuperTuxKart tracks autonomously using Proximal Policy Optimization (PPO). 

This project uses the `pystk2` library to interface with the SuperTuxKart engine, training a neural network through continuous control to steer, accelerate, and brake.

---

## Features
* **Proximal Policy Optimization (PPO):** Implements a state-of-the-art PPO algorithm with Generalized Advantage Estimation (GAE) for highly stable and efficient policy updates.
* **Actor-Critic Architecture:** Uses dual neural networks (Actor for action sampling, Critic for state-value estimation) with 256-dimensional hidden layers and ELU activations.
* **Continuous Control:** Navigates a highly dynamic 3D racing environment using continuous action spaces for smooth Steering, Acceleration, and Braking.
* **Parallel Rollouts:** Utilizes Python's `multiprocessing` module to run multiple environment instances simultaneously, massively accelerating data collection and training.
* **Robust Reward Engineering:** Custom reward function prioritizing track progress, penalizing stalling, maintaining heading alignment, and preventing exploits (like track wrap-around reversing).

---

## Project Structure
* `Environment.ipynb`: The main training loop. Spawns parallel worker processes, collects state transitions, and performs PPO updates. Automatically saves the best-performing models.
* `Inference.ipynb`: The visual inference script. Loads the best trained `.pth` weights and renders the kart driving in real-time at ~50 FPS.
* `actor.py`: Defines the PyTorch `ActorNetwork` which takes in the 52-dimensional state vector and outputs action distributions.
* `critic.py`: Defines the PyTorch `CriticNetwork` which estimates the value of a given state.
* `env_worker.py`: Contains the `SingleInstance` and `ProcessState` classes. This handles all interaction with the `pystk2` API, state observation normalization, and reward math.

---

## State & Action Spaces

### Observation Space (52 Dimensions)
The environment observations are stacked across 4 frames to provide temporal context (velocity over time). Each frame consists of 13 normalized features:
- **Track Error (Dot/Cross):** Distance from the center of the track (absolute and signed).
- **Velocity:** 3D velocity vector of the kart.
- **Heading Alignment:** Dot and cross products of the kart's forward vector against the track's directional vector.
- **Kart Status:** Jump state, rotation quaternions, and normalized distance down the track.

### Action Space (3 Dimensions)
- **Steer:** `[-1.0, 1.0]` (Left to Right)
- **Accelerate:** `[0.0, 1.0]`
- **Brake:** `[0.0, 1.0]`

---

## Getting Started

### Prerequisites
* Python 3.10+
* [SuperTuxKart 1.5](https://supertuxkart.net/Download) installed on your system.
* PyTorch (CPU)

### Installation
1. Clone this repository.
2. Create a virtual environment and install the dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install torch numpy pystk2 jupyter
   ```
3. **Windows Users:** Ensure the path to your SuperTuxKart installation is correct in the scripts (defaults to `C:\Program Files\SuperTuxKart 1.5`).

### Training the Model
To train the model from scratch or resume training:
1. Open `Environment.ipynb` using Jupyter.
2. Run all cells. The script will spawn multiple worker processes and begin PPO training.
3. The weights will automatically be saved as `best_actor.pth` and `best_critic.pth` whenever a new high score is achieved.

### Watching the Bot Drive (Inference)
To watch your trained model drive:
1. Ensure `best_actor.pth` and `best_critic.pth` are in the root directory.
2. Open `Inference.ipynb`.
3. Run all cells. A graphical window will open showing the bot navigating the track.

---

## How it Works
1. **Data Collection:** The main process communicates with parallel workers via `multiprocessing.Pipe`. Each worker runs a headless instance of SuperTuxKart, taking actions and returning states/rewards.
2. **Advantage Estimation:** Once enough transitions are gathered, GAE calculates how much better or worse an action was compared to the Critic's prediction.
3. **Policy Update:** The Actor network is updated to make "good" actions more likely, while the Critic network is updated to better predict future rewards. The PPO clipping function ensures the policy doesn't change too drastically in a single step, preventing collapse.
