# Stack Smart: Reinforcement Learning for Dynamic Block Stacking

## Project Description
This project implements a reinforcement learning agent capable of stacking blocks in a physics-based environment. The agent is trained using Proximal Policy Optimization (PPO) across multiple difficulty stages, adapting to increasingly complex scenarios.

## Installation
1. Clone this repository or download the project files.
2. Create a virtual environment:
   ```bash
   python -m venv venv
3. Activate the virtual environment:

    Windows:
     venv\Scripts\activate
    macOS/Linux:
    source venv/bin/activate

4. Install dependencies:
   pip install -r requirements.txt

Requirements
  Python 3.11.9
  stable-baselines3 2.4.0
  gymnasium 1.0.0
  pybullet 3.2.5
  numpy 1.24.1
  Project Structure
  train_agent.py: Main training script.
  block_stacking_env.py: Custom block stacking environment.
  requirements.txt: List of required packages.

Running the Code
   To train the agent, run the following command:
   python train_agent.py
 Training Stages
   The agent is trained across four difficulty stages:

   Stage 1: Max blocks = 20, Timesteps = 7,500
   Stage 2: Max blocks = 30, Timesteps = 10,000
   Stage 3: Max blocks = 50, Timesteps = 15,000
   Stage 4: Max blocks = 55, Timesteps = 20,000
Results Overview
  The agent shows significant improvement across stages:

  Stage 1: Mean reward ~4352.78 ± 9.33
  Stage 2: Mean reward ~12238.54 ± 14.46
  Stage 3: Mean reward ~49796.70 ± 19.57
 Stage 4: Mean reward ~63480.57 ± 23.99

Author
  Naga Venkata Siva Havinash Reddy Arumalla

