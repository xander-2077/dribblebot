import time
from collections import deque
import copy
import os
from tqdm import tqdm

import torch

# from ml_logger import logger
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook

from params_proto import PrefixProto

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from dribblebot_learn.ppo_cse.metrics_caches import SlotCache, DistCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = "RMA"
    num_steps_per_env = 24  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 400  # check for potential saves every this many iterations
    save_video_interval = 100
    log_freq = 10

    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
    resume_curriculum = True
    resume_checkpoint = "ac_weights_last.pt"


class Runner:

    def __init__(self, env, device="cpu"):
        from .ppo import PPO

        self.device = device
        self.env = env

        actor_critic = ActorCritic(
            self.env.num_obs,
            self.env.num_privileged_obs,
            self.env.num_obs_history,
            self.env.num_actions,
        ).to(self.device)
        # Load weights from checkpoint
        if RunnerArgs.resume:
            # body = wandb.restore(RunnerArgs.resume_checkpoint, run_path=RunnerArgs.resume_path)
            # actor_critic.load_state_dict(torch.load(body.name))
            # print(f"Successfully loaded weights from checkpoint ({RunnerArgs.resume_checkpoint}) and run path ({RunnerArgs.resume_path})")
            actor_critic.load_state_dict(torch.load(RunnerArgs.resume_checkpoint))

        self.alg = PPO(actor_critic, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # init storage and model
        self.alg.init_storage(
            self.env.num_train_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_obs_history],
            [self.env.num_actions],
        )

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = -RunnerArgs.save_video_interval

        self.env.reset()

    def learn(
        self,
        num_learning_iterations,
        init_at_random_ep_len=False,
        eval_freq=100,
        curriculum_dump_freq=500,
        eval_expert=False,
    ):
        trigger_sync = TriggerWandbSyncHook()
        wandb.watch(self.alg.actor_critic, log="all", log_freq=RunnerArgs.log_freq)

        if init_at_random_ep_len:
            self.env.episode_length_buf[:] = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # split train and test envs
        num_train_envs = self.env.num_train_envs

        obs_dict = (
            self.env.get_observations()
        )
        obs, privileged_obs, obs_history = (
            obs_dict["obs"],
            obs_dict["privileged_obs"],
            obs_dict["obs_history"],
        )
        obs, privileged_obs, obs_history = (
            obs.to(self.device),
            privileged_obs.to(self.device),
            obs_history.to(self.device),
        )
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in tqdm(range(self.current_learning_iteration, tot_iter)):
            # for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(
                        obs[:num_train_envs],
                        privileged_obs[:num_train_envs],
                        obs_history[:num_train_envs],
                    )

                    ret = self.env.step(actions)
                    obs_dict, rewards, dones, infos = ret
                    obs, privileged_obs, obs_history = (
                        obs_dict["obs"],
                        obs_dict["privileged_obs"],
                        obs_dict["obs_history"],
                    )

                    obs, privileged_obs, obs_history, rewards, dones = (
                        obs.to(self.device),
                        privileged_obs.to(self.device),
                        obs_history.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(
                        rewards[:num_train_envs], dones[:num_train_envs], infos
                    )
                    
                    """
                    train_episode:
                    command_area_nominal: 1.0
                    'max_action': tensor(0., device='cuda:0'),
                    'max_command_bound': tensor(0., device='cuda:0'),
                    'max_command_duration': tensor(0.5000, device='cuda:0'),
                    'max_command_freq': tensor(3., device='cuda:0'),
                    'max_command_offset': tensor(0., device='cuda:0'),
                    'max_command_phase': tensor(0.5000, device='cuda:0'),
                    'max_command_swing_height': tensor(0.0900, device='cuda:0'),
                    'max_command_x_vel': tensor(1.4989, device='cuda:0'),
                    'max_command_y_vel': tensor(1.4989, device='cuda:0'),
                    'max_command_yaw_vel': tensor(0., device='cuda:0'),
                    'min_action': tensor(0., device='cuda:0'),
                    'min_command_bound': tensor(0., device='cuda:0'),
                    'min_command_duration': tensor(0.5000, device='cuda:0'),
                    'min_command_freq': tensor(3., device='cuda:0'),
                    'min_command_offset': tensor(0., device='cuda:0'),
                    'min_command_phase': tensor(0.5000, device='cuda:0'),
                    'min_command_swing_height': tensor(0.0900, device='cuda:0'),
                    'min_command_x_vel': tensor(-1.4996, device='cuda:0'),
                    'min_command_y_vel': tensor(-1.4967, device='cuda:0'),
                    'min_command_yaw_vel': tensor(0., device='cuda:0'),
                    ------
                    'rew_action_rate': tensor(0., device='cuda:0'),
                    'rew_action_smoothness_1': tensor(0., device='cuda:0'),
                    'rew_action_smoothness_2': tensor(0., device='cuda:0'),
                    'rew_collision': tensor(0., device='cuda:0'),
                    'rew_dof_acc': tensor(0., device='cuda:0'),
                    'rew_dof_pos': tensor(0., device='cuda:0'),
                    'rew_dof_pos_limits': tensor(0., device='cuda:0'),
                    'rew_dof_vel': tensor(0., device='cuda:0'),
                    'rew_dribbling_ball_vel': tensor(0., device='cuda:0'),
                    'rew_dribbling_ball_vel_angle': tensor(0., device='cuda:0'),
                    'rew_dribbling_ball_vel_norm': tensor(0., device='cuda:0'),
                    'rew_dribbling_robot_ball_pos': tensor(0., device='cuda:0'),
                    'rew_dribbling_robot_ball_vel': tensor(0., device='cuda:0'),
                    'rew_dribbling_robot_ball_yaw': tensor(0., device='cuda:0'),
                    'rew_orientation': tensor(0., device='cuda:0'),
                    'rew_torques': tensor(0., device='cuda:0'),
                    'rew_total': tensor(0., device='cuda:0'),
                    'rew_tracking_contacts_shaped_force': tensor(0., device='cuda:0'),
                    'rew_tracking_contacts_shaped_vel': tensor(0., device='cuda:0')}
                    """
                    if "train/episode" in infos:
                        for key, value in infos["train/episode"].items():
                            if "rew" in key:
                                wandb.log({f'reward/{key}': value}, step=it)
                            else:
                                wandb.log({f'command/{key}': value}, step=it)
                        
                        # wandb.log({f'train_episode/{key}': value for key, value in infos["train/episode"].items()}, step=it)

                    if "curriculum" in infos:

                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(
                            cur_reward_sum[new_ids_train].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids_train].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                    if "curriculum/distribution" in infos:
                        distribution = infos["curriculum/distribution"]

                self.alg.compute_returns(
                    obs_history[:num_train_envs], privileged_obs[:num_train_envs]
                )

            (
                mean_value_loss,
                mean_surrogate_loss,
                mean_adaptation_module_loss,
                mean_decoder_loss,
                mean_decoder_loss_student,
                mean_adaptation_module_test_loss,
                mean_decoder_test_loss,
                mean_decoder_test_loss_student,
                mean_adaptation_losses_dict,
            ) = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            wandb.log(
                {
                    "update_info/time_iter": learn_time,
                    # "time_iter": logger.split('epoch'),
                    "update_info/adaptation_loss": mean_adaptation_module_loss,
                    "update_info/mean_value_loss": mean_value_loss,
                    "update_info/mean_surrogate_loss": mean_surrogate_loss,
                    "update_info/mean_decoder_loss": mean_decoder_loss,
                    "update_info/mean_decoder_loss_student": mean_decoder_loss_student,
                    "update_info/mean_decoder_test_loss": mean_decoder_test_loss,
                    "update_info/mean_decoder_test_loss_student": mean_decoder_test_loss_student,
                    "update_info/mean_adaptation_module_test_loss": mean_adaptation_module_test_loss,
                },
                step=it,
            )

            if RunnerArgs.save_video_interval:
                self.log_video(it)

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs

            wandb.log({"timesteps": self.tot_timesteps, "iterations": it}, step=it)
            trigger_sync()

            if it % RunnerArgs.save_interval == 0:
                print(f"Saving model at iteration {it}")

                path = "./tmp/legged_data"
                os.makedirs(path, exist_ok=True)

                ac_weight_path = f"{path}/ac_weights_{it}.pt"
                torch.save(self.alg.actor_critic.state_dict(), ac_weight_path)
                wandb.save(ac_weight_path)

                ac_weight_path = f"{path}/ac_weights_latest.pt"
                torch.save(self.alg.actor_critic.state_dict(), ac_weight_path)
                wandb.save(ac_weight_path)

                adaptation_module_path = f"{path}/adaptation_module_{it}.jit"
                adaptation_module = copy.deepcopy(
                    self.alg.actor_critic.adaptation_module
                ).to("cpu")
                traced_script_adaptation_module = torch.jit.script(adaptation_module)
                traced_script_adaptation_module.save(adaptation_module_path)

                adaptation_module_path = f"{path}/adaptation_module_latest.jit"
                traced_script_adaptation_module.save(adaptation_module_path)

                body_path = f"{path}/body_{it}.jit"
                body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to("cpu")
                traced_script_body_module = torch.jit.script(body_model)
                traced_script_body_module.save(body_path)

                body_path = f"{path}/body_latest.jit"
                traced_script_body_module.save(body_path)

                # logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
                # logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)

                ac_weights_path = f"{path}/ac_weights_{it}.pt"
                torch.save(self.alg.actor_critic.state_dict(), ac_weights_path)
                ac_weights_path = f"{path}/ac_weights_latest.pt"
                torch.save(self.alg.actor_critic.state_dict(), ac_weights_path)

                wandb.save(f"./tmp/legged_data/adaptation_module_{it}.jit")
                wandb.save(f"./tmp/legged_data/body_{it}.jit")
                wandb.save(f"./tmp/legged_data/ac_weights_{it}.pt")
                wandb.save(f"./tmp/legged_data/adaptation_module_latest.jit")
                wandb.save(f"./tmp/legged_data/body_latest.jit")
                wandb.save(f"./tmp/legged_data/ac_weights_latest.pt")

            self.current_learning_iteration += num_learning_iterations

        path = "./tmp/legged_data"

        os.makedirs(path, exist_ok=True)

    def log_video(self, it):
        if it - self.last_recording_it >= RunnerArgs.save_video_interval:
            self.env.start_recording()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            import numpy as np

            video_array = (
                np.concatenate(
                    [np.expand_dims(frame, axis=0) for frame in frames], axis=0
                )
                .swapaxes(1, 3)
                .swapaxes(2, 3)
            )
            print(video_array.shape)
            # logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)
            wandb.log({"video": wandb.Video(video_array, fps=1 / self.env.dt)}, step=it)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert
