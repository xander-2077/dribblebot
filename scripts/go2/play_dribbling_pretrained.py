import isaacgym
assert isaacgym
import torch
import numpy as np
import glob, os
from tqdm import tqdm
from matplotlib import pyplot as plt
from dribblebot.envs import *
from dribblebot.envs.base.legged_robot_config import Cfg
from dribblebot.envs.go2.go2_config import config_go2
from dribblebot.envs.go2.velocity_tracking import VelocityTrackingEasyEnv


def load_policy(logdir):
    body_path = glob.glob(os.path.join(logdir, 'body*'))[0]
    adaptation_module_path = glob.glob(os.path.join(logdir, 'adaptation_module*'))[0]
    body = torch.jit.load(body_path, map_location="cpu")
    adaptation_module = torch.jit.load(adaptation_module_path, map_location='cpu')

    def policy(obs, info={}):
        """
        obs: 
        - obs: shape (num_envs, 75)
        - obs_history: shape (num_envs, 1125)   15步历史观测 75*15=1125
        [consisting of the 15-step history of command ct, ball position bt, joint positions and velocities qt, d qt, gravity unit vector in the body frame gt, global body yaw ψt, and timing reference variables θ cmd t. 
        The commands ct consist of the target ball velocities v cmd x, v cmd y in the global frame]
        - privileged_obs: shape (num_envs, 6)

        obs["obs_history"]: shape (num_envs, 1125)
        latent: shape (num_envs, 6)
        action: shape (num_envs, 12)
        """
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, headless=False):
    dirs = glob.glob(f"../../runs/{label}/*")
    logdir = sorted(dirs)[-1]
    
    import yaml
    with open(logdir + "/config.yaml", 'rb') as file: 
        cfg = yaml.safe_load(file)
        cfg = cfg["Cfg"]

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.randomize_tile_roughness = True
    # Cfg.domain_rand.tile_roughness_range = [0.1, 0.1]
    Cfg.domain_rand.tile_roughness_range = [0.0, 0.0]

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    # Cfg.env.num_observations = 75
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.num_border_boxes = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.robot.name = "go2"
    Cfg.sensors.sensor_names = [
                        "ObjectSensor",         # 3 [0:3] z==0
                        "OrientationSensor",    # 3 [3:6]
                        "RCSensor",             # 15 [6:21]
                        "JointPositionSensor",  # 12 [21:33]
                        "JointVelocitySensor",  # 12 [33:45]
                        "ActionSensor",         # 12 [45:57]
                        "ActionSensor",         # 12 [57:69]
                        "ClockSensor",          # 4 [69:73]
                        "YawSensor",            # 1 [73:74]
                        "TimingSensor",         # 1 [74:75]
                        ]
    Cfg.sensors.sensor_args = {
                        "ObjectSensor": {},
                        "OrientationSensor": {},
                        "RCSensor": {},
                        "JointPositionSensor": {},
                        "JointVelocitySensor": {},
                        "ActionSensor": {},
                        "ActionSensor": {"delay": 1},
                        "ClockSensor": {},
                        "YawSensor": {},
                        "TimingSensor":{},
                        }

    Cfg.sensors.privileged_sensor_names = {
                        "BodyVelocitySensor": {},
                        "ObjectVelocitySensor": {},
    }
    Cfg.sensors.privileged_sensor_args = {
                        "BodyVelocitySensor": {},
                        "ObjectVelocitySensor": {},
    }
    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = False
    Cfg.control.control_type = "P"  # TODO: CHANGE CONTROL TYPE actuator_net or P
    Cfg.env.num_privileged_obs = 6
    
    # viewer
    Cfg.viewer.lookat = [9, 9, 1]
    Cfg.viewer.pos = [5, 5, 3]
    
    # import inspect, os
    # Cfg_source = inspect.getsource(Cfg)
    # config_path = os.path.join("play_config_cls", "config.py")
    # with open(config_path, "w") as f:
    #     f.write("# Come from play_dribbling_pretrained.py to store config of playing.\n")
    #     f.write("from params_proto import PrefixProto, ParamsProto\n\n")
    #     f.write(Cfg_source)
        
    from dribblebot.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    policy = load_policy(logdir)
    return env, policy

def log_to_file(obs, action, filename="record.txt", mode="a"):
    obs = obs.squeeze().cpu().numpy()
    action = action.squeeze().cpu().numpy()
    sensor_info = {
        "ObjectSensor": obs[0:3],
        "OrientationSensor": obs[3:6],
        "RCSensor": obs[6:21],
        "JointPositionSensor": obs[21:33],
        "JointVelocitySensor": obs[33:45],
        "ActionSensor": obs[45:57],
        "ActionSensor_last": obs[57:69],
        "ClockSensor": obs[69:73],
        "YawSensor": obs[73:74],
        "TimingSensor": obs[74:75],
    }
    import os 
    file_path = os.path.join("record", filename)
    with open(file_path, mode) as file:
        for sensor_name, sensor_values in sensor_info.items():
            file.write(f"{sensor_name}: {sensor_values}\n")
        file.write("-" * 20 + "\n")
        file.write(f"Action: {action}\n")
        file.write("-" * 40 + "\n")
            
            
def play_go2(headless=True, use_joystick=False, plot=False):

    # label = "improbableailab/dribbling/bvggoq26"
    # label = "xander2077/dribbling/0bzdzy6s"
    # label = "xander2077/dribbling/smdr6ns9"
    # label = "xander2077/dribbling/cdmgbim9"
    # label = "xander2077/dribbling/wks8c7nc"
    # label = "xander2077/dribbling/2o9rndfb"
    label = "xander2077/dribbling/v5uq6hjm"
    
    env, policy = load_env(label, headless=headless)
    num_eval_steps = 500  # default: 5000
    
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}
    
    if use_joystick:
        import pygame
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            raise Exception("No joystick detected")
        else:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            
        axis_id = {
            "LX": 0,  # Left stick axis x
            "LY": 1,  # Left stick axis y
            "RX": 3,  # Right stick axis x
            "RY": 4,  # Right stick axis y
            "LT": 2,  # Left trigger
            "RT": 5,  # Right trigger
            "DX": 6,  # Directional pad x
            "DY": 7,  # Directional pad y
        }
        button_id = {
            "X": 2,
            "Y": 3,
            "B": 1,
            "A": 0,
            "LB": 4,
            "RB": 5,
            "SELECT": 6,
            "START": 7,
        }
    
    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, -1.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.09
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.0
    stance_length_cmd = 0.0
    aux_reward_cmd = 0.0

    # record variables
    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    joint_positions = np.zeros((num_eval_steps, 12))
    
    action_list = np.zeros((num_eval_steps, 12))
    
    # # if to use the stored observation
    # store_obs = np.load("record/store_obs.npy")
    # store_obs_tensor = torch.tensor(store_obs).reshape(500, -1).to("cuda:0")    

    obs = env.reset()
    ep_rew = 0
    for i in tqdm(range(num_eval_steps)):
        # # If to use the stored observation
        # obs["obs_history"] = store_obs_tensor[i].unsqueeze(0)
        
        with torch.no_grad():
            actions = policy(obs)

        if use_joystick:
            pygame.event.get()
            x_vel_cmd = joystick.get_axis(axis_id["LX"])
            y_vel_cmd = -joystick.get_axis(axis_id["LY"])
        
        env.commands[:, 0] = x_vel_cmd              #  * 2
        env.commands[:, 1] = y_vel_cmd              #  * 2
        env.commands[:, 2] = yaw_vel_cmd            #  * 0.25
        env.commands[:, 3] = body_height_cmd        # 0.0 * 2
        env.commands[:, 4] = step_frequency_cmd     # 3.0
        env.commands[:, 5:8] = gait                 # [0.5, 0, 0]
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd   # 0.09 * 0.15
        env.commands[:, 10] = pitch_cmd             # 0.0 * 0.3
        env.commands[:, 11] = roll_cmd              # 0.0 * 0.3
        env.commands[:, 12] = stance_width_cmd      # 0.0
        env.commands[:, 13] = stance_length_cmd     # 0.0
        env.commands[:, 14] = aux_reward_cmd        # 0.0
        
        obs, rew, done, info = env.step(actions)
        ep_rew += rew

        # if i == 0:
        #     log_to_file(obs["obs"], actions, mode="w")
        # else:
        #     log_to_file(obs["obs"], actions, mode="a")
        
        if plot:    
            measured_x_vels[i] = env.base_lin_vel[0, 0]
            joint_positions[i] = obs["obs"].squeeze()[21:33].cpu()
            action_list[i] = actions[0]
            
            out_of_limits = -(env.dof_pos - env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
            out_of_limits += (env.dof_pos - env.dof_pos_limits[:, 1]).clip(min=0.)  # upper limit
        
    if plot:
        default_joint_order = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
        ]
        
        plot_order = [
            'FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint',
            'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint',
            'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint'
        ]
        
        default_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5
        }
        default_joint_pos = np.zeros_like(action_list)
        for k, v in default_joint_angles.items():
            default_joint_pos[:, default_joint_order.index(k)] = v

        action_list = action_list * 0.25
        action_list[:, [0, 3, 6, 9]] *= 0.5
        
        joint_pos_target = np.zeros_like(action_list)
        joint_pos_target = action_list + default_joint_pos
        
        joint_positions = joint_positions + default_joint_pos

        time = np.linspace(0, num_eval_steps * env.dt, num_eval_steps)
        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        fig.suptitle("Action and Joint Position for All 12 Joints", fontsize=16)

        for i in range(12):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            
            joint_name = plot_order[i]
            actual_index = default_joint_order.index(joint_name)

            # ax.plot(time, action_list[:, actual_index], linestyle="-", label="Action", color="b")
            # ax.plot(time, joint_positions[:, actual_index], linestyle="--", label="Joint Position", color="r")        
            ax.plot(time, joint_pos_target[:, actual_index], linestyle="--", label="Joint Pos Target", color="r")
            ax.plot(time, joint_positions[:, actual_index], linestyle="-", label="Joint Position", color="b")

            default_angle = default_joint_angles[joint_name]
            ax.axhline(y=default_angle, color='g', linestyle=':', label='Default Position')

            ax.set_title(joint_name.rstrip('_joint').replace('_', ' '))
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    
    # # save data
    # np.save("./record/action_list", action_list, allow_pickle=False)
    # np.save("./record/joint_positions", joint_positions, allow_pickle=False)
    
    if plot:
        # plot target and measured forward velocity
        fig, axs = plt.subplots(2, 1, figsize=(12, 5))
        axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
        axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
        axs[0].legend()
        axs[0].set_title("Forward Linear Velocity")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Velocity (m/s)")

        axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
        axs[1].set_title("Joint Positions")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Joint Position (rad)")

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    play_go2(headless=False, use_joystick=False, plot=True)
