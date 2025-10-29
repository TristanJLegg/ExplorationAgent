# ExplorationAgent

ExplorationAgent is a reinforcement learning research platform focused on maximising discovery in large, complex 3D environments.  
The environments often contain hidden rooms and unseen content, and the agent’s objective is to discover as much of the world as possible within each episode.  

The project explores different sequence modelling approaches for exploration, from classic recurrent networks (GRUs) to transformer-based architectures inspired by GPT. The aim is to enable agents not only to act, but also to build memory, plan, and push deeper into unknown spaces using intrinsic curiosity signals.  

This work formed part of my Master’s thesis: “Maximising Exploration and Discovery in Unknown Large Intricate 3D Worlds”.  
Key contributions include:

- **Multiple Architectures:** Implementations of GRU, feed-forward, and transformer-based agents for exploration.  
- **Curiosity-Driven Learning:** Agents are guided by intrinsic motivation to encourage discovery of new areas and hidden rooms, not only extrinsic rewards.  
- **Evaluation Framework:** Tools for systematic training, benchmarking, and visualisation through TensorBoard and video output.  
- **Research to Practice:** Designed for extensibility, enabling rapid prototyping and testing of new exploration strategies.  

This repository demonstrates how sequence models can improve reinforcement learning agents’ ability to generalise and operate in expansive, partially observable environments.

## Models

This "main" branch contains the implementations of the GRU and non-sequential models of this work. Swap to the "transformer" branch to use the transformer architecture.

## Installation

These instructions were tested using a bash terminal in Ubuntu 22.04.4 LTS.

Clone this repository to your development folder and navigate in:
```bash
git clone https://github.com/TristanJLegg/ExplorationAgent.git &&
cd ExplorationAgent
```

A [Miniconda](https://docs.anaconda.com/miniconda/install/) environment is created and activated with Python 3.10.14 to manage our packages:
```bash
conda create -n ExplorationAgent python=3.10.14 &&
conda activate ExplorationAgent
```

Install the required pip packages:
```bash
pip install -r requirements.txt
```

## Scripts

*All scripts must be run from within the repository directory*.

Edit the config.yaml to change the training, environment and output parameters.

To start training run:
```bash
python train.py configs/train_gru_config.yaml
```
This will create a training tensorboard in the 'runs' directory.

To evaluate the trained agent run:
```bash
python evaluate.py configs/evaluate_gru_config.yaml
```
This will create a tensorboard in the 'runs' directory that will store the results of 10 environments running in parallel.

To take a video of the agent playing the environment run:
```bash
python video.py configs/video_gru_config.yaml
```
This will create a video of an environment episode in the 'videos' directory.

To play and test the environment yourself run:
```bash
python play_world.py
```

## Troubleshooting

Ensure these system packages are installed on your system:
```bash
apt install libgl1-mesa-glx libglu1-mesa libglfw3-dev libgles2-mesa-dev libfreetype6 libfreetype6-dev
```

Ensure these conda packages are installed in your Miniconda environment:
```bash
conda install -c conda-forge libstdcxx-ng
```
