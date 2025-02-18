import datetime

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from miniworld.hidden_miniworld import HiddenMiniWorldEnv, Room
from miniworld.entity import Agent

from src.parameters import EnvironmentParameters
from src.world.map import Map
from src.world.camera import point_cloud_from_depth_map, get_camera_intrinsics, get_camera_extrinsics
from src.world.preset_room_objects import *

class HiddenRoomWorld(HiddenMiniWorldEnv):
    def __init__(
            self,
            environment_parameters: EnvironmentParameters,
            map_rooms: list[list[Room]],
            **kwargs
        ) -> None:
        self.generated_map_rooms = map_rooms

        self.num_full_rooms = (len(self.generated_map_rooms) * len(self.generated_map_rooms[0]))
        self.num_hidden_rooms = environment_parameters.num_hidden_rooms

        # Room size
        self.size = environment_parameters.room_size

        # Rewards
        self.coverage_for_termination = environment_parameters.coverage_for_termination
        self.step_penalty = environment_parameters.step_penalty
        # Agent
        self.depth_cutoff = environment_parameters.depth_cutoff

        # Doors
        self.door_width = environment_parameters.door_width
        self.door_height = environment_parameters.door_height
        self.door_length = environment_parameters.door_length

        # Randomize seed if not set in kwargs
        if environment_parameters.seed is not None:
            self.seed = environment_parameters.seed
        else:
            self.seed = np.random.randint(0, 100000)

        np.random.seed(self.seed)

        # Map
        self.map = Map(
            environment_parameters, 
            map_bounds=(0, self.size * len(map_rooms) + self.door_length * (len(map_rooms) - 1))
        )
        self.explorable_area = self.get_explorable_area(map_rooms)

        super(HiddenRoomWorld, self).__init__(self, **kwargs)

        # Override the max episode steps
        self.max_episode_steps = environment_parameters.max_episode_steps

        # Define the action space as turn_left, turn_right, move_forward
        self.action_space = spaces.Discrete(3)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.step_count += 1

        fwd_step = 0.25 # 0.25 meters
        fwd_drift = 0 # no drift
        turn_step = 15 # 15 degrees

        if action == self.actions.move_forward:
            self.move_agent(fwd_step, fwd_drift) 
        elif action == self.actions.turn_left:
            self.turn_agent(turn_step)
        elif action == self.actions.turn_right:
            self.turn_agent(-turn_step)

        # Determine if the agent is in a specific room
        pos = self.agent.pos
        room_x = int(pos[0] // (self.size + self.door_length))
        room_z = int(pos[2] // (self.size + self.door_length))

        if self.visited_rooms[room_x][room_z] == 0:
            in_room = (
                self.map_rooms[room_x][room_z].min_x <= pos[0] <= self.map_rooms[room_x][room_z].max_x and 
                self.map_rooms[room_x][room_z].min_z <= pos[2] <= self.map_rooms[room_x][room_z].max_z
            )
            if in_room:
                self.visited_rooms[room_x][room_z] = 1
                self.num_visited_rooms += 1

                if self.generated_map_rooms[room_x][room_z].type == "hidden":
                    self.num_visited_hidden_rooms += 1

        observation = self.render_obs()

        termination = False
        truncation = False

        depth_map = self.render_depth()

        camera_intrinsics = get_camera_intrinsics(depth_map.shape[0], depth_map.shape[1], 60)
        camera_extrinsics = get_camera_extrinsics(self.agent.cam_pos, self.agent.cam_dir)

        step_point_cloud = point_cloud_from_depth_map(depth_map, camera_intrinsics, camera_extrinsics, depth_cutoff=self.depth_cutoff)
        reward, new_seen_area, completed, coverage = self.calculate_reward(step_point_cloud)

        large_location_map, small_location_map = self.map.get_maps(self.agent.cam_pos, self.agent.cam_dir)

        observation = (observation, large_location_map, small_location_map)

        if self.step_count >= self.max_episode_steps:
            termination = False
            truncation = True
            reward = 0

        if completed:
            termination = True
            truncation = False

        info = {
            'coverage': coverage,
            'visited_rooms': (self.num_visited_rooms / self.num_full_rooms),
            'visited_hidden_rooms': (self.num_visited_hidden_rooms / self.num_hidden_rooms)
        }

        return observation, reward, termination, truncation, info
    
    def reset(self, **kwargs):
        # Remove current agent and place a new one
        self.entities.remove(self.agent)
        self.place_agent()

        # Visited room data
        self.visited_rooms = np.zeros((len(self.map_rooms), len(self.map_rooms[0])))
        self.num_visited_rooms = 0
        self.num_visited_hidden_rooms = 0

        observation, reward = super(HiddenRoomWorld, self).reset(**kwargs)

        self.map.reset()
        self.seen_area = 0

        large_location_map, small_location_map = self.map.get_maps(self.agent.cam_pos, self.agent.cam_dir)

        observation = (observation, large_location_map, small_location_map)
        reward = 0

        return observation, reward

    def get_top_map(self):
        assert self.render_mode == 'rgb_array', 'Only rgb_array is supported.'

        render_map = np.zeros((self.map.top_map.shape[0], self.map.top_map.shape[1], 3), dtype=np.uint8)
        render_map[self.map.top_map == 0] = np.array([255, 255, 255]) # white
        render_map[self.map.top_map == 1] = np.array([0, 0, 255]) # red
        render_map[self.map.top_map == 2] = np.array([0, 255, 0]) # green

        return render_map
    
    def calculate_reward(self, point_cloud: np.ndarray) -> float:
        # Calculate reward based on coverage and step penalty
        new_seen_area = self.map.get_seen_area(point_cloud)
        area_reward = new_seen_area - self.seen_area

        area_reward_scaling = 1 / ((self.map.top_map.shape[0] * self.map.top_map.shape[1]) * self.explorable_area)
        reward = area_reward * area_reward_scaling - self.step_penalty

        completed = new_seen_area * area_reward_scaling > self.coverage_for_termination

        # Set the new seen area
        self.seen_area = new_seen_area
        coverage = new_seen_area * area_reward_scaling
         
        return reward, new_seen_area, completed, coverage
    
    def place_door_between_rooms(self, room_a: Room, room_b: Room, door_width: float = 1.5, door_height: float = 2.1) -> None:
        # Check if either room is hidden
        hidden = room_a.type == "hidden" or room_b.type == "hidden"

        # Check if door should be placed on the x or z axis
        if room_a.min_x == room_b.min_x:
            if not hidden:
                self.connect_rooms(
                    room_a,
                    room_b,
                    min_x=((room_a.min_x) + (self.size / 2) - (door_width / 2)),
                    max_x=((room_a.min_x) + (self.size / 2) + (door_width / 2)),
                    max_y=door_height
                )
            else:
                self.connect_rooms_hidden(
                    room_a,
                    room_b,
                    min_x=((room_a.min_x) + (self.size / 2) - (door_width / 2)),
                    max_x=((room_a.min_x) + (self.size / 2) + (door_width / 2)),
                    max_y=door_height
                )
        elif room_a.min_z == room_b.min_z:
            if not hidden:
                self.connect_rooms(
                    room_a,
                    room_b,
                    min_z=((room_a.min_z) + (self.size / 2) - (door_width / 2)),
                    max_z=((room_a.min_z) + (self.size / 2) + (door_width / 2)),
                    max_y=door_height
                )
            else:
                self.connect_rooms_hidden(
                    room_a,
                    room_b,
                    min_z=((room_a.min_z) + (self.size / 2) - (door_width / 2)),
                    max_z=((room_a.min_z) + (self.size / 2) + (door_width / 2)),
                    max_y=door_height
                )
        else:
            raise ValueError("Rooms are not adjacent") 

    def _gen_world(self) -> None:
        # Create map_rooms and hidden_rooms
        self.map_rooms = [[None]*len(self.generated_map_rooms[0]) for _ in range(len(self.generated_map_rooms))]
        self.hidden_rooms = []

        # Create the rooms
        for x in range(len(self.map_rooms)):
            for z in range(len(self.map_rooms[0])):
                room = self.generated_map_rooms[x][z]
                if room is not None:
                    self.map_rooms[x][z] = self.create_room(room.type, room.min_x, room.max_x, room.min_z, room.max_z, orientation=room.orientation)

        # Place doors between rooms
        for x in range(len(self.map_rooms)):
            for z in range(len(self.map_rooms[0])):
                room = self.generated_map_rooms[x][z]
                if room is not None:
                    if room.doors[0]:
                        self.place_door_between_rooms(
                            self.map_rooms[x][z], 
                            self.map_rooms[x+1][z], 
                            door_width=self.door_width, 
                            door_height=self.door_height
                        )
                    if room.doors[1]:
                        self.place_door_between_rooms(
                            self.map_rooms[x][z], 
                            self.map_rooms[x][z+1], 
                            door_width=self.door_width, 
                            door_height=self.door_height
                        )

        # Texture the rooms
        for row in self.map_rooms:
            for room in row:
                if room is not None:
                    self.texture_room(room)

        # Place objects in the rooms
        for row in self.map_rooms:
            for room in row:
                if room is not None:
                    self.place_objects(room, orientation=room.orientation)

    def texture_room(self, room: Room) -> None:
        if room.type == "kitchen":
            room.wall_tex_name = "drywall"
            room.floor_tex_name = "wood"
            room.ceil_tex_name = "drywall"
        elif room.type == "bathroom":
            room.wall_tex_name = "ceiling_tiles"
            room.floor_tex_name = "ceiling_tiles"
            room.ceil_tex_name = "ceiling_tiles"
        elif room.type == "lounge":
            room.wall_tex_name = "drywall"
            room.floor_tex_name = "wood_lounge"
            room.ceil_tex_name = "drywall"
        elif room.type == "bedroom":
            room.wall_tex_name = "drywall"
            room.floor_tex_name = "redcarpet"
            room.ceil_tex_name = "drywall"
        elif room.type == "office":
            room.wall_tex_name = "drywall"
            room.floor_tex_name = "wood"
            room.ceil_tex_name = "drywall"
        elif room.type == "dining":
            room.wall_tex_name = "drywall"
            room.floor_tex_name = "wood"
            room.ceil_tex_name = "drywall"
        elif room.type == "laundry":
            room.wall_tex_name = "cinder_blocks"
            room.floor_tex_name = "concrete_tiles"
            room.ceil_tex_name = "cinder_blocks"

    def place_objects(self, room: Room, orientation: int = 0) -> None:
        objects = []
        
        if room.type == "kitchen":
            objects = get_kitchen_objects(
                room_size=self.size, 
                room_orientation=orientation, 
                room_position=[room.min_x, 0, room.min_z]
            )
        elif room.type == "bathroom":
            objects = get_bathroom_objects(
                room_size=self.size, 
                room_orientation=orientation, 
                room_position=[room.min_x, 0, room.min_z]
            )
        elif room.type == "lounge":
            objects = get_lounge_objects(
                room_size=self.size, 
                room_orientation=orientation, 
                room_position=[room.min_x, 0, room.min_z]
            )
        elif room.type == "bedroom":
            objects = get_bedroom_objects(
                room_size=self.size, 
                room_orientation=orientation, 
                room_position=[room.min_x, 0, room.min_z]
            )
        elif room.type == "office":
            objects = get_office_objects(
                room_size=self.size, 
                room_orientation=orientation, 
                room_position=[room.min_x, 0, room.min_z]
            )
        elif room.type == "dining":
            objects = get_dining_objects(
                room_size=self.size, 
                room_orientation=orientation, 
                room_position=[room.min_x, 0, room.min_z]
            )
        elif room.type == "laundry":
            objects = get_laundry_objects(
                room_size=self.size, 
                room_orientation=orientation, 
                room_position=[room.min_x, 0, room.min_z]
            )

        for object in objects:
            self.place_entity(
                object["mesh"], 
                pos=object["position"],
                dir=object["direction"]
            )

    def create_room(self, room_type: str, min_x: float, max_x: float, min_z: float, max_z: float, orientation: int=0) -> Room:
        if room_type == "kitchen":
            room = self.add_kitchen_room(min_x, max_x, min_z, max_z, orientation)
        elif room_type == "bathroom":
            room = self.add_bathroom_room(min_x, max_x, min_z, max_z, orientation)
        elif room_type == "lounge":
            room = self.add_lounge_room(min_x, max_x, min_z, max_z, orientation)
        elif room_type == "bedroom":
            room = self.add_bedroom_room(min_x, max_x, min_z, max_z, orientation)
        elif room_type == "office":
            room = self.add_office_room(min_x, max_x, min_z, max_z, orientation)
        elif room_type == "dining":
            room = self.add_dining_room(min_x, max_x, min_z, max_z, orientation)
        elif room_type == "laundry":
            room = self.add_laundry_room(min_x, max_x, min_z, max_z, orientation)
        elif room_type == "hidden":
            room = self.add_hidden_room(min_x, max_x, min_z, max_z, orientation)
        return room

    def add_kitchen_room(self, min_x: float, max_x: float, min_z: float, max_z: float, orientation: int=0) -> Room:
        room = self.add_rect_room(min_x, max_x, min_z, max_z)
        room.type = "kitchen"
        room.orientation = orientation
        return room
    
    def add_bathroom_room(self, min_x: float, max_x: float, min_z: float, max_z: float, orientation: int=0) -> Room:
        room = self.add_rect_room(min_x, max_x, min_z, max_z)
        room.type = "bathroom"
        room.orientation = orientation
        return room
    
    def add_lounge_room(self, min_x: float, max_x: float, min_z: float, max_z: float, orientation: int=0) -> Room:
        room = self.add_rect_room(min_x, max_x, min_z, max_z)
        room.type = "lounge"
        room.orientation = orientation
        return room
    
    def add_bedroom_room(self, min_x: float, max_x: float, min_z: float, max_z: float, orientation: int=0) -> Room:
        room = self.add_rect_room(min_x, max_x, min_z, max_z)
        room.type = "bedroom"
        room.orientation = orientation
        return room
    
    def add_office_room(self, min_x: float, max_x: float, min_z: float, max_z: float, orientation: int=0) -> Room:
        room = self.add_rect_room(min_x, max_x, min_z, max_z)
        room.type = "office"
        room.orientation = orientation
        return room
    
    def add_dining_room(self, min_x: float, max_x: float, min_z: float, max_z: float, orientation: int=0) -> Room:
        room = self.add_rect_room(min_x, max_x, min_z, max_z)
        room.type = "dining"
        room.orientation = orientation
        return room
    
    def add_laundry_room(self, min_x: float, max_x: float, min_z: float, max_z: float, orientation: int=0) -> Room:
        room = self.add_rect_room(min_x, max_x, min_z, max_z)
        room.type = "laundry"
        room.orientation = orientation
        return room
    
    def add_hidden_room(self, min_x: float, max_x: float, min_z: float, max_z: float, orientation: int=0) -> Room:
        room = self.add_rect_room(min_x, max_x, min_z, max_z)
        room.type = "hidden"
        room.orientation = orientation
        self.hidden_rooms.append(room)
        return room
    
    def get_explorable_area(self, map_rooms) -> float:
        explorable_blocks = 0
        for row in map_rooms:
            for room in row:
                if room is not None:
                    explorable_blocks += self.size * self.size
                    if room.doors[0]:
                        explorable_blocks += self.door_length * self.door_width
                    if room.doors[1]:
                        explorable_blocks += self.door_length * self.door_width

        total_room_area = len(map_rooms) * len(map_rooms[0]) * self.size * self.size
        total_horizontal_door_area = len(map_rooms) * (len(map_rooms[0]) - 1) * self.door_length * self.door_width
        total_vertical_door_area = (len(map_rooms) - 1) * len(map_rooms[0]) * self.door_length * self.door_width
        total_area = total_room_area + total_horizontal_door_area + total_vertical_door_area

        return explorable_blocks / total_area