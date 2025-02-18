import os
import datetime

import torch

from src.world.generation import Room_Placeholder
from src.parameters import Parameters, Hyperparameters, EnvironmentParameters

def save_model(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        seed: int,
        map_rooms: list[Room_Placeholder],
        parameters: Parameters,
        hyperparameters: Hyperparameters,
        environment_parameters: EnvironmentParameters,
        path: str
    ):
    # Create models folder if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'seed': seed,
            'map_rooms': map_rooms,
            'parameters': parameters,
            'hyperparameters': hyperparameters,
            'environment_parameters': environment_parameters
        },
        path
    )

def load_model(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        path: str
    ):
    checkpoint = torch.load(path, weights_only=False)
    if model:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    seed = checkpoint['seed']
    map_rooms = checkpoint['map_rooms']
    config_info = {
        'parameters': checkpoint['parameters'],
        'hyperparameters': checkpoint['hyperparameters'],
        'environment_parameters': checkpoint['environment_parameters']
    }

    return model, optimizer, step, seed, map_rooms, config_info

def get_name_without_path(path: str):
    return path.split("/")[-1]