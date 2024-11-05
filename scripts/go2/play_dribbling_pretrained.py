import isaacgym

assert isaacgym
import torch
import numpy as np
import glob, os

from dribblebot.envs import *
from dribblebot.envs.base.legged_robot_config import Cfg
from dribblebot.envs.go2.go2_config import config_go2
from dribblebot.envs.go2.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm

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
        i = 0
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
                        "YawSensor",            # 1 [73:74] 部署时需要积分
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
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"
    Cfg.env.num_privileged_obs = 6
    
    import inspect, os
    Cfg_source = inspect.getsource(Cfg)
    config_path = os.path.join("play_config_cls", "config.py")
    with open(config_path, "w") as f:
        f.write("from params_proto import PrefixProto, ParamsProto\n\n")
        f.write(Cfg_source)
        
    from dribblebot.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    policy = load_policy(logdir)
    return env, policy


def play_go2(headless=True):

    # label = "improbableailab/dribbling/bvggoq26"
    label = "xander2077/dribbling/0bzdzy6s"
    # label = "xander2077/dribbling/smdr6ns9"
    env, policy = load_env(label, headless=headless)

    num_eval_steps = 500  # 本地测试时，可以设置为5000
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.09
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.0

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    joint_positions = np.zeros((num_eval_steps, 12))
    action_list = np.zeros((num_eval_steps, 12))

    # import imageio
    # mp4_writer = imageio.get_writer('dribbling.mp4', fps=50)

    def save_observation_to_file(obs, filename="observation_output.txt", mode="a"):
        obs = obs.squeeze().cpu().numpy()
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
            "TimingSensor": obs[74:75]
        }
        import os
        file_path = os.path.join("record", filename)
        with open(file_path, mode) as file:
            for sensor_name, sensor_values in sensor_info.items():
                file.write(f"{sensor_name}: {sensor_values}\n")
            file.write("-" * 40 + "\n")

    obs = env.reset()
    ep_rew = 0
    for i in tqdm(range(num_eval_steps)):
        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0] = x_vel_cmd              # 0.0 * 2
        env.commands[:, 1] = y_vel_cmd              # -1.0 * 2
        env.commands[:, 2] = yaw_vel_cmd            # 0.0 * 0.25
        env.commands[:, 3] = body_height_cmd        # 0.0 * 2
        env.commands[:, 4] = step_frequency_cmd     # 3.0
        env.commands[:, 5:8] = gait                 # [0.5, 0, 0]
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd   # 0.09 * 0.15
        env.commands[:, 10] = pitch_cmd             # 0.0 * 0.3
        env.commands[:, 11] = roll_cmd              # 0.0 * 0.3
        env.commands[:, 12] = stance_width_cmd      # 0.0
        obs, rew, done, info = env.step(actions)
        if i == 0:
            save_observation_to_file(obs["obs"], mode="w")
        else:
            save_observation_to_file(obs["obs"], mode="a")
        measured_x_vels[i] = env.base_lin_vel[0, 0]
        joint_positions[i] = env.dof_pos[0, :].cpu()
        action_list[i] = actions[0].cpu()
        ep_rew += rew

        img = env.render(mode='rgb_array')
        # mp4_writer.append_data(img)

        out_of_limits = -(env.dof_pos - env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (env.dof_pos - env.dof_pos_limits[:, 1]).clip(min=0.)

    # mp4_writer.close()
    
    action_list = action_list * 0.25
    np.save("action_list", action_list, allow_pickle=False)
    np.save("joint_positions", joint_positions, allow_pickle=False)
    
    # plot target and measured forward velocity
    from matplotlib import pyplot as plt
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
    play_go2(headless=False)
