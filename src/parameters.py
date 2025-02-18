import yaml

class Parameters:
    def __init__(
            self,
            model,
            model_save_interval,
            console_update_interval,
            training_comment,
            training_tensorboard_directory
        ) -> None:
            self.model = model
            self.model_save_interval = model_save_interval
            self.console_update_interval = console_update_interval
            self.training_comment = training_comment
            self.training_tensorboard_directory = training_tensorboard_directory

    def __str__(self) -> str:
        return f"{{ \n\
            Parameters: model={self.model}, \n\
            model_save_interval={self.model_save_interval}, \n\
            console_update_interval={self.console_update_interval}, \n\
            training_comment={self.training_comment} \n\
            training_tensorboard_directory={self.training_tensorboard_directory} \n\
        }}"

class Hyperparameters:
    def __init__(
        self,
        num_steps,
        num_processes,
        num_mini_batch,
        rollout_size,
        gamma,
        learning_rate,
        ppo_epochs,
        clip,
        value_loss_coef,
        entropy_coef,
        max_grad_norm,
        eps,
        hidden_state_size,
        sequential_model
    ) -> None:
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.num_mini_batch = num_mini_batch
        self.rollout_size = rollout_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.ppo_epochs = ppo_epochs
        self.clip = clip
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.eps = eps
        self.hidden_state_size = hidden_state_size
        self.sequential_model = sequential_model

    def __str__(self) -> str:
        return f"{{ \n\
            num_steps={self.num_steps}, \n\
            num_processes={self.num_processes}, \n\
            num_mini_batch={self.num_mini_batch}, \n\
            rollout_size={self.rollout_size}, \n\
            gamma={self.gamma}, \n\
            learning_rate={self.learning_rate}, \n\
            ppo_epochs={self.ppo_epochs}, \n\
            clip={self.clip}, \n\
            alue_loss_coef={self.value_loss_coef}, \n\
            entropy_coef={self.entropy_coef}, \n\
            max_grad_norm={self.max_grad_norm}, \n\
            eps={self.eps} \n\
            hidden_state_size={self.hidden_state_size} \n\
            sequential_model={self.sequential_model} \n\
        }}"

class EnvironmentParameters:
    def __init__(
            self,
            observation_type,
            large_map_size,
            small_map_size,
            large_map_range,
            small_map_range,
            grid_blocks_size,
            agent_height,
            agent_base,
            agent_obstacle_height_tolerance,
            obs_height,
            obs_width,
            environment_size,
            num_hidden_rooms,
            minimum_full_rooms,
            room_size,
            door_width,
            door_height,
            door_length,
            max_episode_steps,
            coverage_for_termination,
            step_penalty,
            depth_cutoff,
            seed
    ) -> None:
        self.observation_type = observation_type
        self.large_map_size = large_map_size
        self.small_map_size = small_map_size
        self.large_map_range = large_map_range
        self.small_map_range = small_map_range
        self.grid_blocks_size = grid_blocks_size
        self.agent_height = agent_height
        self.agent_base = agent_base
        self.agent_obstacle_height_tolerance = agent_obstacle_height_tolerance
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.environment_size = environment_size
        self.num_hidden_rooms = num_hidden_rooms
        self.minimum_full_rooms = minimum_full_rooms
        self.room_size = room_size
        self.door_width = door_width
        self.door_height = door_height
        self.door_length = door_length
        self.max_episode_steps = max_episode_steps
        self.coverage_for_termination = coverage_for_termination
        self.step_penalty = step_penalty
        self.depth_cutoff = depth_cutoff
        self.seed = seed

    def __str__(self) -> str:
        return f"{{ \n\
            observation_type={self.observation_type}, \n\
            large_map_size={self.large_map_size}, \n\
            small_map_size={self.small_map_size}, \n\
            large_map_range={self.large_map_range}, \n\
            small_map_range={self.small_map_range}, \n\
            grid_blocks_size={self.grid_blocks_size}, \n\
            agent_height={self.agent_height}, \n\
            agent_base={self.agent_base}, \n\
            agent_obstacle_height_tolerance={self.agent_obstacle_height_tolerance}, \n\
            obs_height={self.obs_height}, \n\
            obs_width={self.obs_width}, \n\
            environment_size={self.environment_size}, \n\
            num_hidden_rooms={self.num_hidden_rooms}, \n\
            minimum_full_rooms={self.minimum_full_rooms}, \n\
            room_size={self.room_size}, \n\
            door_width={self.door_width}, \n\
            door_height={self.door_height}, \n\
            door_length={self.door_length}, \n\
            max_episode_steps={self.max_episode_steps}, \n\
            coverage_for_termination={self.coverage_for_termination}, \n\
            step_penalty={self.step_penalty}, \n\
            depth_cutoff={self.depth_cutoff}, \n\
            seed={self.seed} \n\
        }}" 

class EvaluationParameters:
     def __init__(
            self,
            evaluation_model, 
            evaluation_num_episodes,
            evaluation_board_comment
        ) -> None:
        self.evaluation_model = evaluation_model
        self.evaluation_num_episodes = evaluation_num_episodes
        self.evaluation_board_comment = evaluation_board_comment

class VideoParameters:
     def __init__(
            self,
            video_model,
            video_num_episodes,
            video_fps
        ) -> None:
        self.video_model = video_model
        self.video_num_episodes = video_num_episodes
        self.video_fps = video_fps

def load_parameters(file_path: str) -> Parameters:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    parameters = Parameters(
        model=config["model"],
        model_save_interval=config["model_save_interval"],
        console_update_interval=config["console_update_interval"],
        training_comment=config["training_comment"],
        training_tensorboard_directory=config["training_tensorboard_directory"]
    )

    return parameters

def load_hyperparameters(file_path: str) -> Hyperparameters:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    hyperparameters = Hyperparameters(
        num_steps=config["num_steps"],
        num_processes=config["num_processes"],
        num_mini_batch=config["num_mini_batch"],
        rollout_size=config["rollout_size"],
        gamma=config["gamma"],
        learning_rate=config["learning_rate"],
        ppo_epochs=config["ppo_epochs"],
        clip=config["clip"],
        value_loss_coef=config["value_loss_coef"],
        entropy_coef=config["entropy_coef"],
        max_grad_norm=config["max_grad_norm"],
        eps=config["eps"],
        hidden_state_size=config["hidden_state_size"],
        sequential_model=config["sequential_model"]
    )

    return hyperparameters

def load_environment_parameters(file_path: str) -> EnvironmentParameters:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    environment_parameters = EnvironmentParameters(
        observation_type=config["observation_type"],
        large_map_size=config["large_map_size"],
        small_map_size=config["small_map_size"],
        large_map_range=config["large_map_range"],
        small_map_range=config["small_map_range"],
        grid_blocks_size=config["grid_blocks_size"],
        agent_height=config["agent_height"],
        agent_base=config["agent_base"],
        agent_obstacle_height_tolerance=config["agent_obstacle_height_tolerance"],
        obs_height=config["obs_height"],
        obs_width=config["obs_width"],
        environment_size=config["environment_size"],
        num_hidden_rooms=config["num_hidden_rooms"],
        minimum_full_rooms=config["minimum_full_rooms"],
        room_size=config["room_size"],
        door_width=config["door_width"],
        door_height=config["door_height"],
        door_length=config["door_length"],
        max_episode_steps=config["max_episode_steps"],
        coverage_for_termination=config["coverage_for_termination"],
        step_penalty=config["step_penalty"],
        depth_cutoff=config["depth_cutoff"],
        seed=config["seed"]
    )

    return environment_parameters

def load_evaluation_parameters(file_path: str) -> EvaluationParameters:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    evaluation_parameters = EvaluationParameters(
        evaluation_model=config["evaluation_model"],
        evaluation_num_episodes=config["evaluation_num_episodes"],
        evaluation_board_comment=config["evaluation_board_comment"]
    )

    return evaluation_parameters

def load_video_parameters(file_path: str) -> VideoParameters:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    video_parameters = VideoParameters(
        video_model=config["video_model"],
        video_num_episodes=config["video_num_episodes"],
        video_fps=config["video_fps"]
    )

    return video_parameters