import math

import numpy as np
from miniworld.entity import MeshEnt
    
def get_kitchen_objects(
        room_size: int = 6, 
        room_orientation: int = 0, 
        room_position: list[float] = [0, 0, 0]
    ) -> list[dict[MeshEnt, list[float], int]]:
    # Entraces at 1, 2, 3
    return [
        create_object_dict("kitchenCabinetCornerInner", [0.52, 0, 0.52], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenCabinetDrawer 1", [0.52, 0, 1.5], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenCoffeeMachine", [0.51, 1, 1.25], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenBlender", [0.51, 1, 1.75], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenFridge", [0.51, 0, 3], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenStove", [0.51, 0, 4.55], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenCabinetDrawer 1", [0.51, 0, 5.5], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenCabinetUpper", [0.51, 1.65, 4.55], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenCabinetUpperDouble", [0.51, 1.65, 5.5], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenCabinetCornerInner", [0.52, 0, 6.48], 0, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenCabinetDrawer 1", [1.5, 0, 6.49], 0, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenMicrowave", [1.5, 1, 6.49], 0, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenCabinet", [5.5, 0, 6.49], 0, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenCabinetCornerInner", [6.48, 0, 6.48], math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenCabinet", [6.49, 0, 5.5], math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenCabinetUpperDouble", [6.49, 1.65, 5.5], math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenCabinetDrawer 1", [6.49, 0, 1.5], math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenCabinetUpperDouble", [6.49, 1.65, 1.5], math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenCabinetCornerInner", [6.48, 0, 0.52], math.pi, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenCabinetDrawer 1", [5.5, 0, 0.51], math.pi, 0.225, room_size, room_orientation, room_position),
        create_object_dict("kitchenSink", [1.5, 0, 0.51], math.pi, 0.225, room_size, room_orientation, room_position),
    ]

def get_bathroom_objects(
        room_size: int = 6,
        room_orientation: int = 0,
        room_position: list[float] = [0, 0, 0]
    ) -> list[dict[MeshEnt, list[float], int]]:
    # Entrance at 2
    return [
        create_object_dict("bathtub", [0.6, 0, 1.2], -math.pi / 2, 0.225, room_size, room_orientation, room_position, collider=rotate_collider([1, 0.75], room_orientation)),
        create_object_dict("shower", [0.7, 0, 6.3], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("toilet", [6, 0, 0.51], math.pi, 0.225, room_size, room_orientation, room_position),
        create_object_dict("bathroomSink", [3.5, 0, 0.51], math.pi, 0.225, room_size, room_orientation, room_position),
    ]

def get_lounge_objects(
        room_size: int = 6,
        room_orientation: int = 0,
        room_position: list[float] = [0, 0, 0]
    ) -> list[dict[MeshEnt, list[float], int]]:
    # Entrances at 0, 1, 3
    return [
        create_object_dict("loungeSofa", [6.5, 0, 3.5], math.pi / 2, 0.225, room_size, room_orientation, room_position, collider=rotate_collider([0.25, 0.75], room_orientation)),
        create_object_dict("loungeChair", [6.25, 0, 1.7], math.pi / 2 + math.pi / 8, 0.225, room_size, room_orientation, room_position),
        create_object_dict("loungeChair", [6.25, 0, 5.3], math.pi / 2 - math.pi / 8, 0.225, room_size, room_orientation, room_position),
        create_object_dict("tableCoffeeGlass", [5, 0, 3.5], math.pi / 2, 0.225, room_size, room_orientation, room_position, collider=rotate_collider([0.5, 0.5], room_orientation)),
        create_object_dict("cabinetTelevision", [3.5, 0, 3.5], -math.pi / 2, 0.225, room_size, room_orientation, room_position, collider=rotate_collider([0.25, 0.75], room_orientation)),
        create_object_dict("televisionModern", [3.5, 0.7, 3.5], -math.pi / 2, 0.225, room_size, room_orientation, room_position, collider=rotate_collider([0.25, 0.75], room_orientation)),
        create_object_dict("speaker", [3.5, 0, 4.5], -math.pi / 2, 0.225, room_size, room_orientation, room_position, rotate_collider([0.25, 0.25], room_orientation)),
        create_object_dict("speaker", [3.5, 0, 2.4], -math.pi / 2, 0.225, room_size, room_orientation, room_position, rotate_collider([0.25, 0.75], room_orientation)),
    ]

def get_bedroom_objects(
        room_size: int = 6,
        room_orientation: int = 0,
        room_position: list[float] = [0, 0, 0]
    ) -> list[dict[MeshEnt, list[float], int]]:
    # Entrances at 0, 1, 3
    return [
        create_object_dict("bedDouble", [5.55, 0, 3.5], math.pi / 2, 0.225, room_size, room_orientation, room_position, collider=rotate_collider([1.75, 1.25], room_orientation)),
        create_object_dict("cabinetBedDrawer", [6.425, 0, 2.1], math.pi / 2, 0.25, room_size, room_orientation, room_position),
        create_object_dict("cabinetBedDrawer", [6.425, 0, 4.9], math.pi / 2, 0.25, room_size, room_orientation, room_position),
        create_object_dict("plantSmall1", [6.325, 0.675, 5.05], 0, 0.225, room_size, room_orientation, room_position),
        create_object_dict("plantSmall2", [6.525, 0.675, 5.05], 0, 0.225, room_size, room_orientation, room_position),
        create_object_dict("plantSmall1", [6.325, 0.675, 1.95], 0, 0.225, room_size, room_orientation, room_position),
        create_object_dict("plantSmall2", [6.525, 0.675, 1.95], 0, 0.225, room_size, room_orientation, room_position),
        create_object_dict("ceilingFan", [5.55, 2.5, 3.6], (3 * math.pi) / 4, 0.25, room_size, room_orientation, room_position, collider=[0, 0]),
        create_object_dict("lampRoundTable", [6.425, 0.675, 2.25], 0, 0.175, room_size, room_orientation, room_position),
        create_object_dict("lampRoundTable", [6.425, 0.675, 4.75], 0, 0.175, room_size, room_orientation, room_position),
        create_object_dict("bookcaseClosedDoors", [1, 0, 6.7], 0, 0.225, room_size, room_orientation, room_position),
        create_object_dict("bookcaseClosedDoors", [0.3, 0, 6], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("loungeChair", [0.75, 0, 0.75], -3 * math.pi / 4, 0.225, room_size, room_orientation, room_position),
    ]

def get_office_objects(
        room_size: int = 6,
        room_orientation: int = 0,
        room_position: list[float] = [0, 0, 0]
    ) -> list[dict[MeshEnt, list[float], int]]:
    # Entrances at 1, 2, 3
    return [
        create_object_dict("desk", [0.5, 0, 3.5], -math.pi / 2, 0.225, room_size, room_orientation, room_position, collider=rotate_collider([0.75, 0.5], room_orientation)),
        create_object_dict("chairDesk", [1.1, 0, 3.2], math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("computerScreen", [0.3, 0.9, 3.5], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("computerKeyboard", [0.7, 0.9, 3.5], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("computerMouse", [0.7, 0.9, 3], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("desk", [0.5, 0, 5.5], -math.pi / 2, 0.225, room_size, room_orientation, room_position, rotate_collider([0.75, 0.5], room_orientation)),
        create_object_dict("chairDesk", [1.1, 0, 5.2], math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("computerScreen", [0.3, 0.9, 5.5], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("computerKeyboard", [0.7, 0.9, 5.5], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("computerMouse", [0.7, 0.9, 5], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("desk", [0.5, 0, 1.5], -math.pi / 2, 0.225, room_size, room_orientation, room_position, rotate_collider([0.75, 0.5], room_orientation)),
        create_object_dict("chairDesk", [1.1, 0, 1.2], math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("computerScreen", [0.3, 0.9, 1.5], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("computerKeyboard", [0.7, 0.9, 1.5], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("computerMouse", [0.7, 0.9, 1], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("cardboardBoxClosed", [6.5, 0, 0.5], math.pi, 0.3, room_size, room_orientation, room_position),
        create_object_dict("cardboardBoxClosed", [6.5, 0, 1.5], math.pi / 4, 0.3, room_size, room_orientation, room_position),
        create_object_dict("cardboardBoxOpen", [5.5, 0, 0.5], -math.pi / 2, 0.3, room_size, room_orientation, room_position),
        create_object_dict("bookcaseOpen", [6.6, 0, 5.875], math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("books", [6.6, 0.325, 5.875], math.pi / 2, 0.3, room_size, room_orientation, room_position),
        create_object_dict("books", [6.6, 0.85, 5.875], math.pi / 2, 0.3, room_size, room_orientation, room_position),
        create_object_dict("books", [6.6, 1.375, 5.875], math.pi / 2, 0.3, room_size, room_orientation, room_position),
    ]

def get_dining_objects(
        room_size: int = 6,
        room_orientation: int = 0,
        room_position: list[float] = [0, 0, 0]
    ) -> list[dict[MeshEnt, list[float], int]]:
    # Entrances at 0, 1, 2, 3
    return [
        create_object_dict("tableCloth", [3.5, 0, 3.5], -math.pi / 2, 0.275, room_size, room_orientation, room_position, rotate_collider([1.5, 0.75], room_orientation)),
        create_object_dict("chairCushion", [3, 0, 3], -math.pi / 2, 0.275, room_size, room_orientation, room_position),
        create_object_dict("chairCushion", [3, 0, 4], -math.pi / 2, 0.275, room_size, room_orientation, room_position),
        create_object_dict("chairCushion", [4, 0, 3], math.pi / 2, 0.275, room_size, room_orientation, room_position),
        create_object_dict("chairCushion", [4, 0, 4], math.pi / 2, 0.275, room_size, room_orientation, room_position),
        create_object_dict("chairCushion", [3.5, 0, 4.55], 0, 0.275, room_size, room_orientation, room_position),
        create_object_dict("chairCushion", [3.5, 0, 2.45], math.pi, 0.275, room_size, room_orientation, room_position),
    ]

def get_laundry_objects(
    room_size: int = 6,
    room_orientation: int = 0,
    room_position: list[float] = [0, 0, 0]
    ) -> list[dict[MeshEnt, list[float], int]]:
    # Entrances at 0, 1, 2, 3
    return [
        create_object_dict("washerDryerStacked", [1.275, 0, 6.5], 0, 0.225, room_size, room_orientation, room_position),
        create_object_dict("washerDryerStacked", [5.725, 0, 6.5], 0, 0.225, room_size, room_orientation, room_position),
        create_object_dict("washerDryerStacked", [6.5, 0, 5.725], math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("washerDryerStacked", [6.5, 0, 1.275], math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("washerDryerStacked", [5.725, 0, 0.5], math.pi, 0.225, room_size, room_orientation, room_position),
        create_object_dict("washerDryerStacked", [1.275, 0, 0.5], math.pi, 0.225, room_size, room_orientation, room_position),
        create_object_dict("washerDryerStacked", [0.5, 0, 1.275], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
        create_object_dict("washerDryerStacked", [0.5, 0, 5.725], -math.pi / 2, 0.225, room_size, room_orientation, room_position),
    ]

def get_hidden_objects(
    room_size: int = 6,
    room_orientation: int = 0,
    room_position: list[float] = [0, 0, 0]
    ) -> list[dict[MeshEnt, list[float], int]]:
    # Entrances at 0, 1, 2, 3
    return []

def create_object_dict(
        name: str,
        position: list[float] = [0, 0, 0],
        direction: float = 0,
        scale: float = 0,
        room_size = 6,
        room_orientation: int = 0, 
        room_position: list[float] = [0, 0, 0],
        collider: list = None
    ) -> dict[MeshEnt, list[float], int]:
    # Create a new object with the given name, position, and direction

    # Rotate Objects around the room based on room_orientation and room_size and set their positions and directions
    # 0: 0 degrees, 1: 90 degrees, 2: 180 degrees, 3: 270 degrees

    # Assert that the orientation is valid
    assert room_orientation in [0, 1, 2, 3], "Orientation must be 0, 1, 2, or 3"

    # Change position type to numpy array
    position = np.array(position)
    room_position = np.array(room_position)
        
    if room_orientation == 0:
        position = position + room_position
    elif room_orientation == 1:
        position = rotate_position_around_room(position, room_size, math.pi / 2) + room_position
        direction = direction + math.pi / 2
    elif room_orientation == 2:
        position = rotate_position_around_room(position, room_size, math.pi) + room_position
        direction = direction + math.pi
    elif room_orientation == 3:
        position = rotate_position_around_room(position, room_size, 3 * math.pi / 2) + room_position
        direction = direction + 3 * math.pi / 2

    new_object = {
        "mesh": MeshEnt(name, scale=scale, collider=collider),
        "position": position,
        "direction": direction
    }

    return new_object
    
def rotate_position_around_room(object_position, room_size, angle: int = 0):
    # Rotate the object's position around the room based on the room's size
    # Assumes object_position has not been offset by the room's position

    room_center = np.array([room_size / 2, 0, room_size / 2])
    
    object_position_centered = np.array(object_position) - room_center
    rotation_matrix = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    rotated_position = np.dot(rotation_matrix, object_position_centered)
    new_position = rotated_position + room_center

    return new_position

def rotate_collider(collider, room_orientation):
    # Rotate the collider around the room based on the room's orientation
    # Assumes collider has not been offset by the room's position

    if room_orientation == 0:
        return collider
    elif room_orientation == 1:
        return [collider[1], collider[0]]
    elif room_orientation == 2:
        return [collider[0], collider[1]]
    elif room_orientation == 3:
        return [collider[1], collider[0]]
    else:
        return collider