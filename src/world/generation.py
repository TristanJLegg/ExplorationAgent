import datetime

import numpy as np

from src.parameters import EnvironmentParameters

class Room_Placeholder:
    def __init__(
            self, 
            room_type: str, 
            min_x: float, 
            max_x: float, 
            min_z: float, 
            max_z: float, 
            orientation: int, 
            doors: list[bool]
        ):
        self.type = room_type
        self.min_x = min_x
        self.max_x = max_x
        self.min_z = min_z
        self.max_z = max_z
        self.orientation = orientation
        # Right Door, Bottom Door
        self.doors = doors

    def set_door(self, door_direction: int):
        self.doors[door_direction] = True

def generate_world(environment_parameters: EnvironmentParameters, set_prints: bool = False) -> list[list[Room_Placeholder]]:
        if set_prints:
            print(f"Generation seed: {environment_parameters.seed}")
            print("Starting world generation...")        

        # Environment parameters
        environment_size = environment_parameters.environment_size
        num_hidden_rooms = environment_parameters.num_hidden_rooms
        minimum_full_rooms = environment_parameters.minimum_full_rooms
        size = environment_parameters.room_size
        door_length = environment_parameters.door_length

        # Initialize the empty map of rooms
        map_rooms = [[None]*environment_size for _ in range(environment_size)]
        hidden_rooms_left = num_hidden_rooms
        empty_room_count = environment_size * environment_size

        # Initialize room_selection
        room_selection = ["kitchen", "bathroom", "lounge", "bedroom", "office", "dining", "laundry"]

        # Regenerate if the number of empty rooms is too high or if there are still hidden rooms left
        while ((1 - minimum_full_rooms) * environment_size * environment_size < empty_room_count) or (hidden_rooms_left > 0):
            # Redeclare the empty map of rooms
            map_rooms = [[None]*environment_size for _ in range(environment_size)]
            hidden_rooms_left = num_hidden_rooms
            empty_room_count = environment_size * environment_size

            # Random Prim's algorithm to connect the rooms
            frontier = []

            # Randomly select and place the first room
            start_x, start_z = np.random.randint(0, environment_size), np.random.randint(0, environment_size)
            room_type, selected_orientation = generate_room(room_selection, 0, 0, empty_room_count)
            map_rooms[start_x][start_z] = Room_Placeholder(
                room_type,
                start_x * size + start_x * door_length, 
                (start_x + 1) * size + start_x * door_length, 
                start_z * size + start_z * door_length, 
                (start_z + 1) * size + start_z * door_length, 
                orientation=selected_orientation,
                doors=[False, False]
            )
            # Update room counts
            empty_room_count -= 1
            # Get valid connections and add to frontier
            valid_connections = get_valid_connections(map_rooms, (start_x, start_z))
            for frontier_connection in valid_connections:
                frontier.append((start_x, start_z, frontier_connection))

            # While there are still rooms to connect
            while len(frontier) > 0:
                # Randomly select a room to connect
                start_x, start_z, connection = frontier[np.random.choice(len(frontier))]
                
                # Check if next room has not been placed
                # Directions 0: North, 1: East, 2: South, 3: West
                if connection == 0:
                    new_x, new_z = start_x - 1, start_z # North
                elif connection == 1:
                    new_x, new_z = start_x, start_z + 1 # East
                elif connection == 2:
                    new_x, new_z = start_x + 1, start_z # South
                elif connection == 3:
                    new_x, new_z = start_x, start_z - 1 # West

                room = map_rooms[new_x][new_z]

                # Only connect if the room has not been placed
                if room is None:
                    # Get room orientation 
                    if connection == 0:
                        new_connection = 2
                    elif connection == 1:
                        new_connection = 3
                    elif connection == 2:
                        new_connection = 0
                    elif connection == 3:
                        new_connection = 1

                    # Initialize room_type, selected_orientation and redeclare valid_connections
                    room_type = None
                    selected_orientation = None
                    valid_connections = []

                    # Regenerate rooms if leading to dead end
                    regenerate_room = True
                    while regenerate_room:
                        room_type, selected_orientation = generate_room(room_selection, new_connection, hidden_rooms_left, empty_room_count)

                        # Place room
                        map_rooms[new_x][new_z] = Room_Placeholder(
                            room_type,
                            new_x * size + new_x * door_length, 
                            (new_x + 1) * size + new_x * door_length, 
                            new_z * size + new_z * door_length, 
                            (new_z + 1) * size + new_z * door_length, 
                            orientation=selected_orientation,
                            doors=[False, False]
                        )

                        valid_connections = get_valid_connections(map_rooms, (new_x, new_z))
                        regenerate_room = (
                            len(valid_connections) == 0 and 
                            len(frontier) == 1 and 
                            exists_empty_adjacent(map_rooms, (new_x, new_z)) and 
                            empty_room_count != hidden_rooms_left
                        )

                    # Update room counts
                    if room_type == "hidden":
                        hidden_rooms_left -= 1
                    empty_room_count -= 1
                    
                    # Place door between rooms
                    place_door_between_rooms(map_rooms, (start_x, start_z), (new_x, new_z))

                    valid_connections = get_valid_connections(map_rooms, (new_x, new_z))
                    for frontier_connection in valid_connections:
                        frontier.append((new_x, new_z, frontier_connection))

                # Remove the current room from the frontier
                frontier.remove((start_x, start_z, connection))

        # Add doors in rooms that have valid connections between them
        for x in range(len(map_rooms)):
            for z in range(len(map_rooms[0])):
                if map_rooms[x][z] is not None:
                    room1_valid_connections = get_valid_connections(map_rooms, (x, z), False, unlink_hidden=False)

                    if not map_rooms[x][z].doors[0] and (x+1) < len(map_rooms[0]):
                        if map_rooms[x+1][z] is not None:
                            room2_valid_connections = get_valid_connections(map_rooms, (x+1, z), False, unlink_hidden=False)
                            if 2 in room1_valid_connections and 0 in room2_valid_connections:
                                place_door_between_rooms(map_rooms, (x, z), (x+1, z))

                    if not map_rooms[x][z].doors[1] and (z+1) < len(map_rooms):
                        if map_rooms[x][z+1] is not None:
                            room2_valid_connections = get_valid_connections(map_rooms, (x, z+1), False, unlink_hidden=False)
                            if 1 in room1_valid_connections and 3 in room2_valid_connections:
                                place_door_between_rooms(map_rooms, (x, z), (x, z+1))

        if set_prints:
            print_map_rooms(map_rooms)
        
        # Return the map of rooms
        return map_rooms

def generate_room(
        room_selection: list[str],
        new_connection: int,
        hidden_rooms_left: int,
        empty_room_count: int
    ):
        # Get random room type
        room_type = weighted_room_selection(room_selection, hidden_rooms_left, empty_room_count)

        # Get random valid orientation
        valid_orientations = get_valid_orientations(room_type, new_connection)
        selected_orientation = np.random.choice(valid_orientations)

        return room_type, selected_orientation
    

def get_valid_orientations(room_type: str, door_direction: int) -> list[int]:
    # Directions 0: North, 1: East, 2: South, 3: West
    valid_orientations = []

    if room_type == "bathroom":
        valid_directions_0 = np.array([2], dtype=int)
    elif room_type == "kitchen":
        valid_directions_0 = np.array([1, 2, 3], dtype=int)
    elif room_type == "lounge":
        valid_directions_0 = np.array([0, 1, 3], dtype=int)
    elif room_type == "bedroom":
        valid_directions_0 = np.array([0, 1, 3], dtype=int)
    elif room_type == "office":
        valid_directions_0 = np.array([1, 2, 3], dtype=int)
    elif room_type == "dining":
        valid_directions_0 = np.array([0, 1, 2, 3], dtype=int)
    elif room_type == "laundry":
        valid_directions_0 = np.array([0, 1, 2, 3], dtype=int)
    elif room_type == "hidden":
        valid_directions_0 = np.array([0, 1, 2, 3], dtype=int)
    
    valid_directions_1 = (valid_directions_0 + 1) % 4
    valid_directions_2 = (valid_directions_0 + 2) % 4
    valid_directions_3 = (valid_directions_0 + 3) % 4

    if door_direction in valid_directions_0:
        valid_orientations.append(0)
    if door_direction in valid_directions_1:
        valid_orientations.append(1)
    if door_direction in valid_directions_2:
        valid_orientations.append(2)
    if door_direction in valid_directions_3:
        valid_orientations.append(3)

    return valid_orientations

def get_valid_connections(map_rooms, coordinates, remove_if_exists: bool=True, unlink_hidden: bool=True) -> list[int]:
    # Get valid room connections
    # Directions 0: North, 1: East, 2: South, 3: West
    x, z = coordinates

    valid_connections = []

    # Remove validity based on the room type
    if map_rooms[x][z].type == "bathroom":
        valid_connections = [2]
    elif map_rooms[x][z].type == "kitchen":
        valid_connections = [1, 2, 3]
    elif map_rooms[x][z].type == "lounge":
        valid_connections = [0, 1, 3]
    elif map_rooms[x][z].type == "bedroom":
        valid_connections = [0, 1, 3]
    elif map_rooms[x][z].type == "office":
        valid_connections = [1, 2, 3]
    elif map_rooms[x][z].type == "dining":
        valid_connections = [0, 1, 2, 3]
    elif map_rooms[x][z].type == "laundry":
        valid_connections = [0, 1, 2, 3]
    elif map_rooms[x][z].type == "hidden":
        if unlink_hidden:
            valid_connections = []
        else:
            valid_connections = [0, 1, 2, 3]

    # rotate using the orientation
    valid_connections = (np.array(valid_connections, dtype=int) + map_rooms[x][z].orientation) % 4

    # Remove validity based on map bounds
    if z == 0:
        valid_connections = np.delete(valid_connections, np.where(valid_connections == 3))
    elif z == len(map_rooms[0]) - 1:
        valid_connections = np.delete(valid_connections, np.where(valid_connections == 1))
    if x == 0:
        valid_connections = np.delete(valid_connections, np.where(valid_connections == 0))
    elif x == len(map_rooms) - 1:
        valid_connections = np.delete(valid_connections, np.where(valid_connections == 2))

    # Remove validity if the room has already been placed
    if remove_if_exists:
        if z + 1 < len(map_rooms[0]):
            if map_rooms[x][z+1] is not None:
                valid_connections = np.delete(valid_connections, np.where(valid_connections == 1))
        if z > 0:
            if map_rooms[x][z-1] is not None:
                valid_connections = np.delete(valid_connections, np.where(valid_connections == 3))
        if x + 1 < len(map_rooms):
            if map_rooms[x+1][z] is not None:
                valid_connections = np.delete(valid_connections, np.where(valid_connections == 2))
        if x > 0:
            if map_rooms[x-1][z] is not None:
                valid_connections = np.delete(valid_connections, np.where(valid_connections == 0))

    return valid_connections

def place_door_between_rooms(
        map_rooms: list[list[Room_Placeholder]], 
        coordinates_a: tuple[int, int], 
        coordinates_b: tuple[int, int]
    ) -> None:
    a_x, a_z = coordinates_a
    b_x, b_z = coordinates_b

    # Check if door should be placed on the x or z axis
    if a_x == b_x:
        if a_z < b_z:
            map_rooms[a_x][a_z].set_door(1)
        elif a_z > b_z:
            map_rooms[b_x][b_z].set_door(1)
    elif a_z == b_z:
        if a_x < b_x:
            map_rooms[a_x][a_z].set_door(0)
        elif a_x > b_x:
            map_rooms[b_x][b_z].set_door(0)
    else:
        raise ValueError("Rooms are not adjacent") 
    
def exists_empty_adjacent(
        map_rooms: list[list[Room_Placeholder]], 
        coordinates: tuple[int, int]
    ) -> bool:
    x, z = coordinates

    if x + 1 < len(map_rooms[0]):
        if map_rooms[x+1][z] is None:
            return True
    if x > 0:
        if map_rooms[x-1][z] is None:
            return True
    if z + 1 < len(map_rooms):
        if map_rooms[x][z+1] is None:
            return True
    if z > 0:
        if map_rooms[x][z-1] is None:
            return True
    
    return False

def weighted_room_selection(
        room_selection: list[str],
        hidden_rooms_left: int,
        empty_room_count: int
    ):
    hidden_room_probability = hidden_rooms_left / empty_room_count

    random_number = np.random.rand()

    if random_number < hidden_room_probability:
        return "hidden"
    
    return np.random.choice(room_selection)

def print_map_rooms(map_rooms: list[list[Room_Placeholder]]) -> None:
        print("Completed map (None = no room):")
        for i in range(len(map_rooms)):
            # Print Room names and horizontal doors
            for j in range(len(map_rooms[0])):
                if map_rooms[i][j] is not None:
                    print("{:^10s}".format(map_rooms[i][j].type), end="")
                    print("{:4s}".format("--" if map_rooms[i][j].doors[1] else ""), end="")
                else:
                    # Spaced for room name and lack of door
                    print("{:^10s}".format("None"), end="")
                    print("{:4s}".format(""), end="")

            # New line
            print()

            # Print vertical doors
            for j in range(len(map_rooms[0])):
                if map_rooms[i][j] is not None:
                    print("{:^10s}".format("|" if map_rooms[i][j].doors[0] else ""), end="")
                    print("{:4s}".format(""), end="")
                else:
                    print("{:14s}".format(""), end="")

            # New line
            print()
        
        print("Completed map orientations (N = no room): ")
        for i in range(len(map_rooms)):
            for j in range(len(map_rooms[0])):
                if map_rooms[i][j] is not None:
                    print("{:<10d}".format(map_rooms[i][j].orientation), end="")
                else:
                    print("{:<10s}".format("N"), end="")
            print()

        # New line
        print()

        print("Completed map valid connections (N = no room, X = not valid): ")
        for i in range(len(map_rooms)):
            for j in range(len(map_rooms[0])):
                if map_rooms[i][j] is not None:
                    valid_connections = get_valid_connections(map_rooms, (i, j), False, False)
                    print("[", end="")
                    for k in range(4):
                        if k in valid_connections:
                            print("{:<2d}".format(k), end="")
                        else:
                            print("{:<2s}".format("X"), end="")
                    print("] ", end="")
                else: 
                    print("[", end="")
                    for k in range(4):
                        print("{:<2s}".format("X"), end="")
                    print("] ", end="")
            print()

        # New line
        print()

def map_rooms_equals(map_rooms_a: list[list[Room_Placeholder]], map_rooms_b: list[list[Room_Placeholder]]) -> bool:
    if len(map_rooms_a) != len(map_rooms_b):
        return False
    if len(map_rooms_a[0]) != len(map_rooms_b[0]):
        return False

    for i in range(len(map_rooms_a)):
        for j in range(len(map_rooms_a[0])):
            if map_rooms_a[i][j] is None and map_rooms_b[i][j] is None:
                continue
            if map_rooms_a[i][j] is None or map_rooms_b[i][j] is None:
                return False
            if map_rooms_a[i][j].type != map_rooms_b[i][j].type:
                return False
            if map_rooms_a[i][j].min_x != map_rooms_b[i][j].min_x:
                return False
            if map_rooms_a[i][j].max_x != map_rooms_b[i][j].max_x:
                return False
            if map_rooms_a[i][j].min_z != map_rooms_b[i][j].min_z:
                return False
            if map_rooms_a[i][j].max_z != map_rooms_b[i][j].max_z:
                return False
            if map_rooms_a[i][j].orientation != map_rooms_b[i][j].orientation:
                return False
            if map_rooms_a[i][j].doors != map_rooms_b[i][j].doors:
                return False
    return True