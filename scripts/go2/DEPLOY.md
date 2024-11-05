# Deployment for Go2


## obs
clip obs:
```
clip_obs = 100.
self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
```

单帧 75 dims, 一共是15帧打包(1125 dims), 收集后过一个 wrapper:
```python
self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
```

obs分解：
```python
"ObjectSensor",         # 3 [0:3]
z坐标置零
在机器狗的局部坐标系中，从机器狗的base坐标到球的坐标
机器狗的base原点到其激光雷达的坐标：
Head_lower: 0.293 0 -0.06
radar: 0.28945 0 -0.046825

"OrientationSensor",    # 3 [3:6]
重力单位向量: projected_gravity 需要除以norm


"RCSensor",             # 15 [6:21]
self.env.commands * self.env.commands_scale
0: x_vel  scale: 2.0
1: y_vel  scale: 2.0
2: yaw_vel  default: 0.0  scale: 0.25
3: body_height  default: 0.0  scale: 2.0
4: step_frequency  default: 3.0  scale: 1.0
5,6,7: gaits = {"pronking": [0, 0, 0],
                "trotting": [0.5, 0, 0],
                "bounding": [0, 0.5, 0],
                "pacing": [0, 0, 0.5]}    default: trotting  scale: 1.0
8: default: 0.5  scale: 1.0
9: footswing_height    default: 0.09  scale: 0.15
10: pitch  default: 0.0  scale: 0.3
11: roll  default: 0.0   scale: 0.3
12: stance_width  default: 0.0   scale: 1.0
# 13和14的数值看起来都非常小
13: stance_length     scale: 1.0
14: aux_reward     scale: 1.0

commands 全零初始化
_resample_commands():



# setting the smaller commands to zero
self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


"JointPositionSensor",  # 12 [21:33]
self.env.dof_pos[:, :self.env.num_actuated_dof] - self.env.default_dof_pos[:, :self.env.num_actuated_dof]

default_dof_pos: [ 0.1000,  0.8000, -1.5000, -0.1000,  0.8000, -1.5000,  0.1000,  1.0000,
                   -1.5000, -0.1000,  1.0000, -1.5000]


"JointVelocitySensor",  # 12 [33:45]
self.env.dof_vel[:, :self.env.num_actuated_dof] * self.env.cfg.obs_scales.dof_vel(0.05)


"ActionSensor",         # 12 [45:57]
self.env.actions


"ActionSensor",         # 12 [57:69]
self.env.last_actions


"ClockSensor",          # 4 [69:73]
self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

foot_indices = [self.gait_indices + phases + offsets + bounds,      +0.5
                self.gait_indices + offsets,                        +0.0
                self.gait_indices + bounds,                         +0.0
                self.gait_indices + phases]                         +0.5

phases = self.commands[:, 5]  ()  (0.5)
offsets = self.commands[:, 6]   (0.0)
bounds = self.commands[:, 7]    (0.0)


"YawSensor",            # 1 [73:74] 
与规定的forward_vec的夹角
部署时使用IMU的rpy即可
机器狗在第一个时间步的局部身体坐标系作为全局参考系
(-pi, pi)


"TimingSensor",         # 1 [74:75]
return self.env.gait_indices.unsqueeze(1)
self.env.gait_indices 初始化为0
迭代:
self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)
self.dt(0.02) = self.cfg.control.decimation(4) * self.sim_params.dt(0.005)
frequencies = self.commands[:, 4] (3.0)
每个step()包含4个仿真步, 即每次拿obs是过了4*0.005=0.02s
部署时需要拿到控制间隔的时间

```



## action

clip action:

```python
clip_actions = 100.
self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
```

计算torque：

```python
self.torques = self._compute_torques(self.actions).view(self.torques.shape)

def _compute_torques(self, actions):
    """ Compute torques from actions.
        Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
        [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

    Args:
        actions (torch.Tensor): Actions

    Returns:
        [torch.Tensor]: Torques sent to the simulation
    """
    # pd controller
    actions_scaled = torch.zeros((actions.shape[0], self.num_dof)).to(self.device)
    actions_scaled[:, :self.num_actuated_dof] = actions[:, :self.num_actuated_dof] * self.cfg.control.action_scale
    if self.num_actions >= 12:
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range

    if self.cfg.domain_rand.randomize_lag_timesteps:
        self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
        self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
    else:
        self.joint_pos_target = actions_scaled + self.default_dof_pos

    control_type = self.cfg.control.control_type

    if control_type == "actuator_net": # this fork
        self.joint_pos_err = self.dof_pos - self.joint_pos_target + self.motor_offsets
        self.joint_vel = self.dof_vel
        torques = self.actuator_network(self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last,
                                        self.joint_vel, self.joint_vel_last, self.joint_vel_last_last)
        self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
        self.joint_pos_err_last = torch.clone(self.joint_pos_err)
        self.joint_vel_last_last = torch.clone(self.joint_vel_last)
        self.joint_vel_last = torch.clone(self.joint_vel)
    elif control_type == "P":
        torques = self.p_gains * self.Kp_factors * (
                self.joint_pos_target - self.dof_pos + self.motor_offsets) - self.d_gains * self.Kd_factors * self.dof_vel
    else:
        raise NameError(f"Unknown controller type: {control_type}")

    torques = torques * self.motor_strengths
    return torch.clip(torques, -self.torque_limits, self.torque_limits)
```

change to:

```python
self.num_dof = 12
self.num_actuated_dof = 12
self.cfg.control.action_scale = 0.25
self.cfg.control.hip_scale_reduction = 0.5
self.cfg.domain_rand.randomize_lag_timesteps = False
self.cfg.domain_rand.lag_timesteps = 6
self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps+1)]
self.default_dof_pos = torch.tensor([ 0.1000,  0.8000, -1.5000, -0.1000,  0.8000, -1.5000,  0.1000,  1.0000, -1.5000, -0.1000,  1.0000, -1.5000], device=self.device)
self.cfg.control.control_type = "P"
self.dof_pos = 拿传感器数据
self.dof_vel = 拿传感器数据
self.p_gains = torch.tensor([20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.], device=self.device)
self.Kp_factors = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], device=self.device)
self.d_gains = torch,tensor([0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000], device=self.device)
self.Kd_factors = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], device=self.device)
self.motor_strengths = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], device=self.device)
self.torque_limits = torch.tensor([23.7000, 23.7000, 45.4300, 23.7000, 23.7000, 45.4300, 23.7000, 23.7000, 45.4300, 23.7000, 23.7000, 45.4300], device=self.device)

self.torques.shape = (self.num_dof, )
self.torques = self._compute_torques(self.actions).view(self.torques.shape)

def _compute_torques(self, actions):
    # pd controller
    actions_scaled = torch.zeros((actions.shape[0], self.num_dof)).to(self.device)
    actions_scaled[:, :self.num_actuated_dof] = actions[:, :self.num_actuated_dof] * self.cfg.control.action_scale
    if self.num_actions >= 12:
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range

    if self.cfg.domain_rand.randomize_lag_timesteps:  # 在play中设为True，部署时可考虑是否使用
        self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
        self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
    else:
        self.joint_pos_target = actions_scaled + self.default_dof_pos

    control_type = self.cfg.control.control_type

    if control_type == "P":
        torques = self.p_gains * self.Kp_factors * (
                self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel

    torques = torques * self.motor_strengths
    return torch.clip(torques, -self.torque_limits, self.torque_limits)
```
joint props:
```
array([( True, -1.0472,  1.0472 , 3, 30.1, 23.7 , 0., 0., 0., 0.),
       ( True, -1.5708,  3.4907 , 3, 30.1, 23.7 , 0., 0., 0., 0.),
       ( True, -2.7227, -0.83776, 3, 15.7, 45.43, 0., 0., 0., 0.),
       ( True, -1.0472,  1.0472 , 3, 30.1, 23.7 , 0., 0., 0., 0.),
       ( True, -1.5708,  3.4907 , 3, 30.1, 23.7 , 0., 0., 0., 0.),
       ( True, -2.7227, -0.83776, 3, 15.7, 45.43, 0., 0., 0., 0.),
       ( True, -1.0472,  1.0472 , 3, 30.1, 23.7 , 0., 0., 0., 0.),
       ( True, -0.5236,  4.5379 , 3, 30.1, 23.7 , 0., 0., 0., 0.),
       ( True, -2.7227, -0.83776, 3, 15.7, 45.43, 0., 0., 0., 0.),
       ( True, -1.0472,  1.0472 , 3, 30.1, 23.7 , 0., 0., 0., 0.),
       ( True, -0.5236,  4.5379 , 3, 30.1, 23.7 , 0., 0., 0., 0.),
       ( True, -2.7227, -0.83776, 3, 15.7, 45.43, 0., 0., 0., 0.)],
      dtype={
        'names': ['hasLimits', 'lower', 'upper', 'driveMode', 'velocity', 'effort', 'stiffness', 'damping', 'friction', 'armature'], 
        'formats': ['?', '<f4', '<f4', '<i4', '<f4', '<f4', '<f4', '<f4', '<f4', '<f4'], 
        'offsets': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36], 
        'itemsize': 40}
     )
```

## Recovery controller

TODO