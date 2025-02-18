import numpy as np

def point_cloud_from_depth_map(depth_map: np.ndarray, camera_intrinsics: np.ndarray, camera_extrinsics: np.ndarray, depth_cutoff: float = 3) -> np.ndarray:
    height = depth_map.shape[0]
    width = depth_map.shape[1]
    inverse_camera_intrinsics = np.linalg.inv(camera_intrinsics)

    img_pixels = np.meshgrid(np.arange(width), np.arange(height))
    img_pixels = np.reshape(img_pixels, (2, -1))
    img_pixels = np.vstack((img_pixels, np.ones((1, img_pixels.shape[1]))))

    camera_to_image_plane = np.dot(inverse_camera_intrinsics, img_pixels)

    depth_map = depth_map.flatten()
    valid = np.logical_and(depth_map > 0, depth_map < depth_cutoff)
    depth_map_valid = depth_map[valid]

    camera_points = np.multiply(camera_to_image_plane[:, valid], depth_map_valid)
    camera_points = np.vstack((camera_points, np.ones((1, camera_points.shape[1]))))

    world_points = np.dot(camera_extrinsics, camera_points)
    world_points = world_points[:3, :].T
    return world_points

def get_camera_intrinsics(height: int, width: int, vertical_fov: float) -> np.ndarray:
    vertical_fov = vertical_fov * np.pi / 180 # convert to radians
    tan_half_vertical_fov = np.tan(vertical_fov / 2)
    # For if the horizontal_fov is not the same as the vertical_fov
    tan_half_horizontal_fov = tan_half_vertical_fov * width / height
    f_x = width / (2 * tan_half_horizontal_fov)
    f_y = height / (2 * tan_half_vertical_fov)
    c_x = width / 2
    c_y = height / 2

    intrinsics = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])

    return intrinsics

def get_camera_extrinsics(position: np.ndarray, camera_direction) -> np.ndarray:
    rotation = _get_camera_rotation(np.array(camera_direction), np.array([0, -1, 0]))

    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = position

    return extrinsics

def _get_camera_rotation(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    right = np.cross(up, forward)
    up = np.cross(forward, right)

    rotation = np.eye(3)
    rotation[:, 0] = right
    rotation[:, 1] = up
    rotation[:, 2] = forward

    return rotation