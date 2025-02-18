import numpy as np
import cv2

from src.parameters import EnvironmentParameters
from src.world.camera import point_cloud_from_depth_map

# Some code is borrowed from https://github.com/taochenshh/exp4nav

class Map:
    def __init__(
            self,
            environment_parameters: EnvironmentParameters,
            map_bounds: tuple[float, float],
        ) -> None:
        # Grid details
        self.grid_blocks_size =  environment_parameters.grid_blocks_size
        assert (map_bounds[1] - map_bounds[0]) % environment_parameters.grid_blocks_size == 0, "Map range must be divisible by grid blocks size."
        self.grid_size = int((map_bounds[1] - map_bounds[0]) // environment_parameters.grid_blocks_size)

        # Map sizes
        self.map_bounds = map_bounds
        self.large_map_range = environment_parameters.large_map_range
        self.small_map_range = environment_parameters.small_map_range
        self.large_map_size = environment_parameters.large_map_size
        self.small_map_size = environment_parameters.small_map_size

        # Agent details
        self.agent_height = environment_parameters.agent_height
        self.agent_base = environment_parameters.agent_base
        self.agent_obstacle_height_tolerance = environment_parameters.agent_obstacle_height_tolerance

    def reset(self) -> None:
        """
        Resets the map.
        """
        self.top_map = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

    def get_seen_area(self, point_cloud) -> tuple[np.ndarray, float]:
        grid_locations = np.floor((point_cloud[:, [0,2]] - self.map_bounds[0]) / self.grid_blocks_size).astype(int)
        grid_matrix = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # Get all obstacles
        high_filter_indices = point_cloud[:, 1] < (self.agent_height - self.agent_obstacle_height_tolerance)
        low_filter_indices = point_cloud[:, 1] > (self.agent_base + self.agent_obstacle_height_tolerance)

        # Safely assign the value 2 to the filtered indices
        self._safe_assign(grid_matrix, grid_locations[high_filter_indices, 0], grid_locations[high_filter_indices, 1], 2)

        # Morphological closing operation
        kernel = np.ones((3, 3), np.uint8)
        grid_matrix = cv2.morphologyEx(grid_matrix, cv2.MORPH_CLOSE, kernel)

        observation_matrix = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        obstacle_indices = np.logical_and(high_filter_indices, low_filter_indices)

        # Safely assign the value 1 to the obstacles
        self._safe_assign(observation_matrix, grid_locations[obstacle_indices, 0], grid_locations[obstacle_indices, 1], 1)

        # Morphological closing operation
        observation_dilation_kernel = 5
        kernel = np.ones((observation_dilation_kernel, observation_dilation_kernel), np.uint8)
        observation_matrix = cv2.morphologyEx(observation_matrix, cv2.MORPH_CLOSE, kernel)

        observation_indices = np.where(observation_matrix == 1)

        # Safely assign the value 1 to the observation indices
        self._safe_assign(grid_matrix, observation_indices[0], observation_indices[1], 1)

        self.top_map[np.where(grid_matrix == 2)] = 2
        self.top_map[np.where(grid_matrix == 1)] = 1
        seen_area = np.sum(self.top_map > 0)
        return seen_area
    
    def get_maps(self, camera_position: np.ndarray, camera_forward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        top_down_map = self.top_map.T.copy()

        # Maybe an issue with grid_size vs map_size?

        half_size = max(top_down_map.shape[0], top_down_map.shape[1], self.large_map_size) * 3

        ego_map = np.zeros((half_size * 2, half_size * 2, 3), dtype=np.uint8) * 255
        location_map = np.zeros((top_down_map.shape[0], top_down_map.shape[1], 3), dtype=np.uint8) * 255

        location_map[top_down_map == 0] = np.array([255, 255, 255]) # white
        location_map[top_down_map == 1] = np.array([0, 0, 255]) # red
        location_map[top_down_map == 2] = np.array([0, 255, 0]) # green

        # Camera grid position
        grid_position = np.floor((camera_position[[2,0]] - self.map_bounds[0]) / self.grid_blocks_size).astype(int)

        x_start = half_size - grid_position[0]
        y_start = half_size - grid_position[1]
        x_end = x_start + top_down_map.shape[0]
        y_end = y_start + top_down_map.shape[1]

        assert x_start >= 0 and y_start >= 0 and x_end <= ego_map.shape[0] and y_end <= ego_map.shape[1], \
        "Camera position is out of bounds."

        ego_map[x_start:x_end, y_start:y_end] = location_map
        center = (half_size, half_size)
        # Is this correct?
        rotation_angle = np.arctan2(camera_forward[2], camera_forward[0]) * 180 / np.pi
        # print(camera_forward)
        # print(rotation_angle)
        rotation_angle = self._constrain_angle(rotation_angle - 90)

        M = cv2.getRotationMatrix2D(center, rotation_angle, 1)
        egocentric_map = cv2.warpAffine(
            ego_map, 
            M, 
            (ego_map.shape[1], ego_map.shape[0]), 
            flags=cv2.INTER_AREA, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(255, 255, 255)
        )

        start = half_size - self.small_map_range
        end = half_size + self.small_map_range
        small_egocentric_map = egocentric_map[start:end, start:end]

        start = half_size - self.large_map_range
        end = half_size + self.large_map_range
        assert start >= 0 and end <= ego_map.shape[0], "Large map size is out of bounds."
        large_egocentric_map = egocentric_map[start:end, start:end]

        # TODO : THIS DOES NOT SEEM RIGHT? DIFFERENT TO THE ORIGINAL PAPER CODE
        # Resize maps to the desired size
        if self.small_map_range * 2 != self.small_map_size:
            small_egocentric_map = cv2.resize(
                small_egocentric_map, 
                (self.small_map_size, self.small_map_size), 
                interpolation=cv2.INTER_AREA
        )
        if self.large_map_range * 2 != self.large_map_size:
            large_egocentric_map = cv2.resize(
                large_egocentric_map, 
                (self.large_map_size, self.large_map_size), 
                interpolation=cv2.INTER_AREA
        )
            
        return large_egocentric_map, small_egocentric_map


    def _constrain_angle(self, angle: float) -> float:
        return (angle + 180) % 360 - 180
    
    def _safe_assign(self, matrix: np.ndarray, x_indices: np.ndarray, y_indices: np.ndarray, value: int) -> None:
        try:
            matrix[x_indices, y_indices] = value
        except IndexError:
            valid_index1 = np.logical_and(x_indices >= 0, x_indices < self.grid_size)
            valid_index2 = np.logical_and(y_indices >= 0, y_indices < self.grid_size)
            valid_index = np.logical_and(valid_index1, valid_index2)
            matrix[x_indices[valid_index], y_indices[valid_index]] = value