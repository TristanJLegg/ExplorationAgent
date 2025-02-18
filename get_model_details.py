import sys

from src.load_and_save import load_model
from src.world.generation import print_map_rooms

if __name__ == "__main__":
    model, optimizer, step, seed, map_rooms, config_info = load_model(None, None, sys.argv[1])
    print(f"Model: {model}")
    print(f"Step: {step}")
    print(f"Seed: {seed}")
    print_map_rooms(map_rooms)
    for key, value in config_info.items():
        print(f"{key}: {value}")