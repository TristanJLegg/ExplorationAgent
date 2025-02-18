# ExplorationAgent

## Models

This "transformer" branch contains the implementation of the decision transformer model of this work. Swap to the "main" branch to use the non-sequential and GRU models. Parts of the GPT implementation are adapted from other sources and attributions are left in the code.

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
python train.py configs/train_dt_config.yaml
```
This will create a training tensorboard in the 'runs' directory.

To evaluate the trained agent run:
```bash
python evaluate.py configs/evaluate_dt_config.yaml
```
This will create a tensorboard in the 'runs' directory that will store the results of 10 environments running in parallel.

To take a video of the agent playing the environment run:
```bash
python video.py configs/video_dt_config.yaml
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