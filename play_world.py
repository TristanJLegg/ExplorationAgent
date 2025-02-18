import os
import sys

import torch
import cv2
import numpy as np
import imageio

from src.world.hidden_room_world import HiddenRoomWorld
from src.world.environments import make_vec_envs
from src.world.generation import generate_world
from src.parameters import load_environment_parameters

def main(**kwargs):
    # Load the environment parameters
    environment_parameters = load_environment_parameters("configs/play_config.yaml")

    if environment_parameters.seed is not None:
        np.random.seed(environment_parameters.seed)
    else:
        environment_parameters.seed = np.random.randint(0, 2**32 - 1)
        np.random.seed(environment_parameters.seed)

    map_rooms = generate_world(environment_parameters, set_prints=True)
    env = make_vec_envs(HiddenRoomWorld, 1, environment_parameters, map_rooms, torch.device("cpu"))
    obs = env.reset()

    print("Welcome to the Hidden Room Exploration Game!")
    print("Actions: a - Turn Left, d - Turn Right, w - Move Forward, p - Save Photo, q - Quit")

    action_dict = {
        49: 0, # 'a'
        52: 1, # 'd'
        71: 2, # 'w'
    }

    # Ask user if they want to record a video of themselves playing the game
    record_video = input("Would you like to record a video of yourself playing the game? (Y/[n]): ")
    if record_video.lower() == "y":
        record_video = True
        # Ask user for the video name
        video_name = input("Please enter the name of the video: ")
        # Ask user for the video frame rate
        video_fps = int(input("Please enter the frame rate (fps) of the video: "))

        # Create videos folder if it does not exist
        if not os.path.exists("./videos"):
            os.makedirs("./videos")
        
        video_writer = imageio.get_writer(f"videos/{video_name}.mp4", fps=video_fps)
    else: 
        record_video = False

    map_type = input("What map type would you like to render? (Top/EgoLarge/EgoSmall,[None]): ").lower()

    user_player_information = input("Would you like to print the player information? (Y/[n]): ")
    if user_player_information.lower() == "y":
        print_agent_information = True
    else:
        print_agent_information = False

    # Check the highest ".png" image number in images folder
    image_count = 0
    for i in range(1, 999):
        if not os.path.isfile(f"./images/{i}.png"):
            image_count = i
            break

    try:
        total_reward = 0
        while True:
            image_render = env.render() 

            place_map = True
            if map_type == "top":
                map_render = env.env_method("get_top_map", indices=[0])[0]
            elif map_type == "egolarge":
                map_render = np.transpose(obs["large_map"], [1, 2, 0])
            elif map_type == "egosmall":
                map_render = np.transpose(obs["small_map"], [1, 2, 0])
            else:
                place_map = False

            if place_map:
                # Resize map to 2x size
                map_render = cv2.resize(
                    map_render, 
                    (map_render.shape[1] * 2, map_render.shape[0] * 2), 
                    interpolation=cv2.INTER_NEAREST
                ) 

                # Place map on top right of image
                image_render[0:map_render.shape[0]:, -map_render.shape[1]:, :] = map_render

            # Append image to video before swapping the red and blue channels
            if record_video:
                video_writer.append_data(image_render)

            # Image render swap red and blue channels
            image_render = cv2.cvtColor(image_render, cv2.COLOR_RGB2BGR)

            cv2.imshow("HiddenRoomWorld", image_render)

            action = cv2.waitKey(0) & 0xFF
            action -= ord('0')

            if action == 66:
                env.reset()
                print("Game reset.")
                continue

            if action == 65:
                print("Quiting game...")
                break

            if action == 64:
                # Create images folder if it does not exist
                if not os.path.exists("./images"):
                    os.makedirs("./images")

                # Save photo
                cv2.imwrite(f"./images/{image_count}.png", image_render)
                print(f"Image {image_count} saved.")
                image_count += 1
                continue
            
            if action not in [49, 52, 71]:
                print("Invalid action. Please enter a, d, w, or p.")
                continue

            action = action_dict[action]

            obs, reward, done, info = env.step(torch.tensor([action]))
            total_reward += reward

            if print_agent_information:
                print(f"player_position: {env.agent.cam_pos}")
                print(f"player_direction: {env.agent.cam_dir}")
                print(f"Reward: {reward}")

            if done:
                print(f"Game Over! Total reward: {reward}")
                print(info)
                break

    except KeyboardInterrupt:
        print("\nGame aborted.")

    finally:
        # Close the environment when the game is finished or aborted
        env.close() 

        print(f"Info: {info}")
        print(f"Total reward: {total_reward}")

        if record_video:
            video_writer.close()
            print(f"Video saved as videos/{video_name}.mp4")

if __name__ == '__main__':
    main(**dict(arg.split('=') for arg in sys.argv[1:])) # kwargs