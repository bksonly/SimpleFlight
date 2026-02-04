# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn import TensorDictModule
import functorch
import numpy as np

from torchrl.data import (
    TensorSpec,
    BoundedTensorSpec,
    UnboundedContinuousTensorSpec as UnboundedTensorSpec,
    CompositeSpec,
    TensorDictReplayBuffer
)
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.objectives.utils import hold_out_net

import copy
from tqdm import tqdm
from omni_drones.utils.torchrl import AgentSpec
from tensordict import TensorDict
from .common import soft_update

class PIDCTBR(nn.Module):
    """
    SimpleFlight (Isaac Sim) 环境下的 PID/CTBR 控制器。
    复现自地面站PID。
    
    预期的 Observation 结构 :
    [0:3]   : x_d - x         (世界坐标系下的位置误差)
    [30:33] : v               (世界坐标系下的线性速度)(此处下标根据future_traj_steps的不同而定)
    [33:42] : R               (旋转矩阵, 机体系到世界系, 展平后的 9 维向量)
    [42:46] : a_{t-1}         (上一时刻的动作，PID 计算中暂未使用)
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = int(action_dim)

        # === 1. 控制器参数 (根据地面站PID硬编码) ===
        self.m = 0.032      # 无人机质量
        self.g = 9.81       # 重力加速度
        self.F_MAX = 0.6134 # 最大推力
        self.T_add = 0.65   # 推力前馈比例 (悬停/偏置)
        
        # 增益参数
        # Kp: [0.045, 0.045, 0.06]
        self.register_buffer("Kp", torch.tensor([0.045, 0.045, 0.06]))
        # Kd: [0.08, 0.08, 0.18]
        self.register_buffer("Kd", torch.tensor([0.08, 0.08, 0.18]))
        
        self.tau = 0.8 # 约化姿态时间常数
        self.theta_max = 20.0 * np.pi / 180.0 # 最大允许倾角 (弧度)
        
        # 幅值限制
        self.max_tilt_force_xy = 0.06      # 水平合力限幅
        self.max_tilt_force_z_up = 0.20    # 上升推力限幅
        self.max_tilt_force_z_down = 0.10  # 下降推力限幅
        self.max_thrust_cmd = 0.75         # 最大下发推力指令限幅
        self.max_rate_deg = 50.0           # 最大角速率限制 (deg/s)

        # 归一化参数 (对应 SimpleFlight Action 空间 [-1, 1])
        # 根据 transforms.py: PIDRate 控制器会将 action * 180.0 得到物理值
        self.norm_rate_scale = 180.0 
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (Batch, Agents, Obs_Dim) 或 (Batch, Obs_Dim)
        Returns:
            归一化后的 CTBR 动作: (..., 4)，范围在 [-1, 1]
            格式: [Rate_Roll, Rate_Pitch, Rate_Yaw, Thrust]
            (与 SimpleFlight 的 PIDRateController 期望输入对齐)
        """
        # 处理 Batch 维度
        original_shape = obs.shape[:-1]
        if obs.ndim > 2:
            obs = obs.reshape(-1, obs.shape[-1]) # 展平 Batch 和 Agents 维度
        
        batch_size = obs.shape[0]
        device = obs.device

        # === 2. 解析观测值 (Observation) ===
        # 位置误差: e_p = p_d - p
        pos_error_scaled = obs[:, 0:3]
        pos_error = pos_error_scaled 

        
        # 速度误差: e_v = v_d - v。假设目标速度 v_d = 0 
        v_curr = obs[:, 30:33]
        vel_error =  -v_curr # 目标速度为 0
        
        # 旋转矩阵 R (机体系 -> 世界系)
        # 将展平的 9 维向量转回 3x3 矩阵(取转置)
        R_flat = obs[:, 33:42]
        R = R_flat.view(batch_size, 3, 3).transpose(1, 2)

        # === 3. 位置环控制 (Position Loop) ===
        # f = Kp * ep + Kd * ev
        f = self.Kp * pos_error + self.Kd * vel_error 
        
        # 合力限幅 
        # 水平 xy 限制在 0.06N
        f_xy_norm = torch.norm(f[:, :2], dim=1, keepdim=True)
        f_xy_scale = torch.clamp(self.max_tilt_force_xy / (f_xy_norm + 1e-6), max=1.0)
        f[:, :2] = f[:, :2] * f_xy_scale
        
        # 垂直 z 限制 (根据地面站PID 上升 0.2N/下降 0.1N)
        f_z = f[:, 2].clone()
        f_z = torch.where(f_z > 0, 
                          torch.clamp(f_z, max=self.max_tilt_force_z_up), 
                          torch.clamp(f_z, min=-self.max_tilt_force_z_down))
        f[:, 2] = f_z
        
        # 计算世界坐标系下的期望推力向量 f_W
        # f_W = [fx, fy, T_add * FMAX + fz]
        f_W = f.clone()
        f_W[:, 2] += self.T_add * self.F_MAX
        
        # === 4. 几何映射与姿态环 (Attitude Loop) ===
        f_W_norm = torch.norm(f_W, dim=1, keepdim=True)
        b_raw = f_W / (f_W_norm + 1e-6)
        
        e3 = torch.tensor([0., 0., 1.], device=device).expand(batch_size, 3)
        
        # 计算当前推力方向与世界 Z 轴的夹角 theta = arccos(b_raw . e3)
        dot_raw_e3 = torch.sum(b_raw * e3, dim=1, keepdim=True)
        theta = torch.acos(torch.clamp(dot_raw_e3, -1.0, 1.0))
        
        # 计算指令姿态轴 b3_cmd (执行倾角超限投影)
        bs_3_cmd = torch.zeros_like(b_raw)
        
        # 情况 1: 夹角在安全范围内
        mask_safe = (theta <= self.theta_max).squeeze()
        bs_3_cmd[mask_safe] = b_raw[mask_safe]
        
        # 情况 2: 超过最大倾角 theta_max，进行锥形投影
        if (~mask_safe).any():
            idxs = (~mask_safe).nonzero(as_tuple=True)[0]
            b_raw_unsafe = b_raw[idxs]
            b_xy = b_raw_unsafe[:, :2]
            b_xy_norm = torch.norm(b_xy, dim=1, keepdim=True)
            scale = torch.sin(torch.tensor(self.theta_max, device=device)) / (b_xy_norm + 1e-6)
            
            bs_3_cmd[idxs, :2] = b_xy * scale
            bs_3_cmd[idxs, 2] = torch.cos(torch.tensor(self.theta_max, device=device))
            
        # 将目标 Z 轴向量投影到机体系: v_B = R^T * b3_cmd
        v_B = torch.bmm(R.transpose(1, 2), bs_3_cmd.unsqueeze(-1)).squeeze(-1)
        
        # 计算约化误差四元数 q_e (从当前机体 Z 轴 e3 到目标 v_B 的最短旋转)
        cross_prod = torch.cross(e3, v_B, dim=1) 
        sin_phi = torch.norm(cross_prod, dim=1, keepdim=True)
        cos_phi = torch.sum(e3 * v_B, dim=1, keepdim=True)
        
        # phi = arccos(e3 . v_B)
        phi = torch.acos(torch.clamp(cos_phi, -1.0, 1.0))
        u = cross_prod / (sin_phi + 1e-6)
        
        q_vec = torch.sin(phi/2) * u
        q_w = torch.cos(phi/2)
        
        # 计算期望机体系角速率 w = (2/tau) * sgn(qw) * q_vec (单位: rad/s)
        sgn_qw = torch.sign(q_w)
        sgn_qw[sgn_qw == 0] = 1.0 # 处理 sign 为 0 的情况
        
        omega_ideal = (2.0 / self.tau) * sgn_qw * q_vec
        
        # === 5. 生成最终控制指令 ===
        
        # -- 推力指令 (Thrust) --
        # b3_cur 是机体 Z 轴在世界系下的朝向 (旋转矩阵 R 的第三列)
        b3_cur = R[:, :, 2]
        
        # f_scalar = max(0, f_W . b3_cur)，如果翻过来了推力置 0
        f_scalar = torch.relu(torch.sum(f_W * b3_cur, dim=1, keepdim=True))
        
        # T_cmd = f_scalar / F_MAX，并限幅在 0.75 内
        T_cmd = f_scalar / self.F_MAX
        T_cmd = torch.clamp(T_cmd, max=self.max_thrust_cmd)
        
        # 归一化: SimpleFlight 的 Action 空间 [-1, 1] 对应 0~1 的推力百分比
        # 映射公式: action = 2 * T_cmd - 1
        action_thrust = 2.0 * T_cmd - 1.0

        # -- 角速率指令 (Rates) --
        # 将弧度制 rad/s 转为角度制 deg/s
        omega_deg = omega_ideal * (180.0 / np.pi)
        
        # 实施限制 (50 deg/s)
        omega_deg = torch.clamp(omega_deg, -self.max_rate_deg, self.max_rate_deg)
        
        # 映射到输出格式
        # 与地面站PID不同，地面站PID约定: [deg(wx), -deg(wy), psi_ref]
        cmd_rate_x = omega_deg[:, 0:1]
        cmd_rate_y = omega_deg[:, 1:2] #(此处和下面的偏航角设置是唯一与地面站PID不一致的地方)
        
        # 偏航 (Yaw): 因为 simpleflight 默认是角速率控制，此处设为 0 来锁定当前航向，实现最简稳健控制
        cmd_rate_z = torch.zeros_like(cmd_rate_x)
        
        # 归一化角速率: transforms.py 中的 scale 为 180.0
        action_rate_x = cmd_rate_x / self.norm_rate_scale
        action_rate_y = cmd_rate_y / self.norm_rate_scale
        action_rate_z = cmd_rate_z / self.norm_rate_scale
        
        # 拼接动作: [Rate_X, Rate_Y, Rate_Z, Thrust]
        base_action = torch.cat([action_rate_x, action_rate_y, action_rate_z, action_thrust], dim=-1)
        
        # 最终截断确保处于 [-1, 1] 范围内
        base_action = torch.clamp(base_action, -1.0, 1.0)

        # 恢复原始维度 (Batch, Agents, 4)
        if len(original_shape) > 1:
            base_action = base_action.view(*original_shape, 4)
            
        return base_action


class SACPolicy(object):

    def __init__(self,
        cfg,
        agent_spec: AgentSpec,
        device: str="cuda",
    ) -> None:
        self.cfg = cfg
        self.agent_spec = agent_spec
        self.device = device

        self.gradient_steps = int(cfg.gradient_steps)
        self.buffer_size = int(cfg.buffer_size)
        self.batch_size = int(cfg.batch_size)

        self.obs_name = ("agents", "observation")
        self.act_name = ("agents", "action")
        self.state_name = ("agents", "state")
        self.reward_name = ("agents", "reward")

        self.make_actor()
        self.make_critic()
        
        self.action_dim = self.agent_spec.action_spec.shape[-1]

        self.base_policy = PIDCTBR(action_dim=self.action_dim).to(self.device)
        # 硬编码 bc loss 系数（不要 YAML）。需要启用时改成 >0。
        self.bc_coef = 1.0

        self.use_base_policy = getattr(cfg, "use_base_policy", False) # 默认不开启，由配置文件sac.yaml决定

        self.target_entropy = - torch.tensor(self.action_dim, device=self.device)
        init_entropy = 1.0
        self.log_alpha = nn.Parameter(torch.tensor(init_entropy, device=self.device).log())
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)

        self.replay_buffer = TensorDictReplayBuffer(
            batch_size=self.batch_size,
            storage=LazyTensorStorage(max_size=self.buffer_size, device='cpu'),
            sampler=RandomSampler(),
        )
    
    def make_actor(self):

        self.policy_in_keys = [self.obs_name]
        self.rl_action_mean_key = ("info", "rl_action_mean")
        self.policy_out_keys = [self.act_name, f"{self.agent_spec.name}.logp", self.rl_action_mean_key]
        
        if self.cfg.share_actor:
            self.actor = TensorDictModule(
                Actor(
                    self.cfg.actor, 
                    self.agent_spec.observation_spec, 
                    self.agent_spec.action_spec
                ),
                in_keys=self.policy_in_keys, out_keys=self.policy_out_keys
            ).to(self.device)
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor.lr)
        else:
            raise NotImplementedError


    def make_critic(self):
        if self.agent_spec.state_spec is not None:
            self.value_in_keys = [self.state_name, self.act_name]
            self.value_out_keys = [f"{self.agent_spec.name}.q"]

            self.critic = Critic(
                self.cfg.critic, 
                1,
                self.agent_spec.state_spec,
                self.agent_spec.action_spec
            ).to(self.device)
        else:
            self.value_in_keys = [self.obs_name, self.act_name]
            self.value_out_keys = [f"{self.agent_spec.name}.q"]

            self.critic = Critic(
                self.cfg.critic, 
                1,
                self.agent_spec.observation_spec,
                self.agent_spec.action_spec
            ).to(self.device)
        
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic.lr)       
        self.critic_loss_fn = {"mse": F.mse_loss, "smooth_l1": F.smooth_l1_loss}[self.cfg.critic_loss]

    def __call__(self, tensordict: TensorDict, deterministic: bool=False) -> TensorDict:
        actor_input = tensordict.select(*self.policy_in_keys)
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]

        # 1) Always compute RL actor action (default env execution path)
        actor_output = self.actor(actor_input)
        tensordict.update(actor_output)

        # 2) Always compute base (model-based) action for BC supervision / debugging
        with torch.no_grad():
            base_action = self.base_policy(actor_input[self.obs_name])
        tensordict.set(("info", "base_action"), base_action)

        # 3) Debug switch: optionally override env-executed action with base_action
        if self.use_base_policy:
            tensordict.set(self.act_name, base_action)

        return tensordict

    def train_op(self, data: TensorDict, verbose: bool=False):
        self.replay_buffer.extend(data.reshape(-1).to('cpu'))

        if len(self.replay_buffer) < 20000:
            print(f"filling buffer: {len(self.replay_buffer)} < {self.cfg.buffer_size}")
            return {}
        
        infos_critic = []
        infos_actor = []

        t = range(1, self.gradient_steps+1)
       
        for gradient_step in tqdm(t) if verbose else t:

            transition = self.replay_buffer.sample()
            transition = transition.to(self.device)
            state   = transition[self.state_name]
            actions = transition[self.act_name]

            reward  = transition[("next", *self.reward_name)]
            next_dones  = transition[("next", "done")].float().unsqueeze(-1)
            next_state  = transition[("next", self.state_name)]

            with torch.no_grad():
                actor_output = self.actor(transition["next"], deterministic=False)
                next_act = actor_output[self.act_name]
                next_logp = actor_output[f"{self.agent_spec.name}.logp"]
                next_qs = self.critic_target(next_state, next_act)
                next_q = torch.min(next_qs, dim=-1, keepdim=True).values
                next_q = next_q - self.log_alpha.exp() * next_logp
                target_q = (reward + self.cfg.gamma * (1 - next_dones) * next_q).detach().squeeze(-1)
                assert not torch.isinf(target_q).any()
                assert not torch.isnan(target_q).any()

            qs = self.critic(state, actions)
            critic_loss = sum(self.critic_loss_fn(q, target_q) for q in qs.unbind(-1))
            self.critic_opt.zero_grad()
            critic_loss.backward()
            critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
            self.critic_opt.step()
            infos_critic.append(TensorDict({
                "critic_loss": critic_loss,
                "critic_grad_norm": critic_grad_norm,
                "q_taken": qs.mean()
            }, []))

            if (gradient_step + 1) % self.cfg.actor_delay == 0:

                with hold_out_net(self.critic):
                    actor_output = self.actor(transition, deterministic=False)
                    act = actor_output[self.act_name]
                    logp = actor_output[f"{self.agent_spec.name}.logp"]
                    act_mean = actor_output[self.rl_action_mean_key]

                    qs = self.critic(state, act)
                    q = torch.min(qs, dim=-1).values
                    awr_loss = None

                    base_action = transition[("info", "base_action")]
                    # ===== AWR (Advantage-Weighted Regression) loss =====
                    # Q(s, a_base)
                    q_base_all = self.critic(state, base_action)
                    q_base = torch.min(q_base_all, dim=-1).values

                    # V(s) ≈ Q(s, a_pi) - alpha * logπ(a_pi|s)
                    # 用rl的当前动作计算v，其实可以多次采样a，但是性能开销会拉大很多，后续如果训练不稳定再用这招
                    alpha = self.log_alpha.exp().detach()
                    v = q - alpha * logp.squeeze(-1)

                    # A(s, a_base)
                    adv = (q_base - v).detach()

                    # weights: exp(adv / beta) with clipping
                    beta = getattr(self, "awr_beta", 0.1)
                    adv_clip = getattr(self, "awr_adv_clip", 10.0)
                    w_clip = getattr(self, "awr_w_clip", 20.0)
                    w = torch.exp(torch.clamp(adv / beta, max=adv_clip)).clamp(max=w_clip)

                    # log π(a_base|s)
                    obs = transition[self.obs_name]
                    actor_net = getattr(self.actor, "module", None) or getattr(self.actor, "_module", None)
                    if actor_net is None or not hasattr(actor_net, "log_prob"):
                        raise RuntimeError("Actor network must expose `log_prob(obs, action)` for AWR.")
                    logp_base = actor_net.log_prob(obs, base_action).squeeze(-1)

                    # weighted NLL
                    awr_loss = -(w * logp_base).mean()

                    #=====BC loss=====
                    bc_loss = F.mse_loss(base_action, act_mean)

                    rl_actor_loss = (self.log_alpha.exp() * logp - q).mean()

                    bc_coef = float(self.bc_coef)
                    actor_loss = (1.0 - bc_coef) * rl_actor_loss + bc_coef * awr_loss
                    # actor_loss = (1.0 - bc_coef) * rl_actor_loss + bc_coef * bc_loss

                    self.actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                    self.actor_opt.step()

                    self.alpha_opt.zero_grad()
                    alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
                    alpha_loss.backward()
                    self.alpha_opt.step()

                    infos_actor.append(TensorDict({
                        "actor_loss": actor_loss,
                        "rl_actor_loss": rl_actor_loss,
                        "actor_grad_norm": actor_grad_norm,
                        "entropy": -logp.mean(),
                        "alpha": self.log_alpha.exp().detach(),
                        "alpha_loss": alpha_loss,
                        **({"awr_loss": awr_loss.detach()} if awr_loss is not None else {}),
                    }, []))


            if (gradient_step + 1) % self.cfg.target_update_interval == 0:
                with torch.no_grad():
                    soft_update(self.critic_target, self.critic, self.cfg.tau)
        
        infos = {**torch.stack(infos_actor), **torch.stack(infos_critic)}
        infos = {k: torch.mean(v).item() for k, v in infos.items()}
        return infos

    def state_dict(self):
        state_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "ctitic_target": self.critic_target.state_dict()
        }
        return state_dict
    
from .modules.networks import MLP
from .modules.distributions import TanhIndependentNormalModule
from .common import make_encoder

class Actor(nn.Module):
    def __init__(self, 
        cfg,
        observation_spec: TensorSpec, 
        action_spec: BoundedTensorSpec,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = make_encoder(cfg, observation_spec)
        
        self.act = TanhIndependentNormalModule(
            self.encoder.output_shape.numel(), 
            action_spec.shape[-1], 
        )

    def forward(self, obs: torch.Tensor, deterministic: bool=False):
        x = self.encoder(obs)
        act_dist = self.act(x)

        if deterministic:
            act = act_dist.mode
        else:
            act = act_dist.rsample()
        log_prob = act_dist.log_prob(act).unsqueeze(-1)
        return act, log_prob, act_dist.mode

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return log π(action|obs) under the current policy distribution."""
        x = self.encoder(obs)
        act_dist = self.act(x)
        return act_dist.log_prob(action).unsqueeze(-1)


class Critic(nn.Module):
    def __init__(self, 
        cfg,
        num_agents: int,
        state_spec: TensorSpec,
        action_spec: BoundedTensorSpec, 
        num_critics: int = 2,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_agents = num_agents
        self.act_space = action_spec
        self.state_space = state_spec
        self.num_critics = num_critics

        self.critics = nn.ModuleList([
            self._make_critic() for _ in range(self.num_critics)
        ])

    def _make_critic(self):
        if isinstance(self.state_space, (BoundedTensorSpec, UnboundedTensorSpec)):
            action_dim = self.act_space.shape[-1]
            state_dim = self.state_space.shape[-1]
            num_units = [
                action_dim * self.num_agents + state_dim, 
                *self.cfg["hidden_units"]
            ]
            base = MLP(num_units)
        else:
            raise NotImplementedError
        
        v_out = nn.Linear(base.output_shape.numel(), 1)
        return nn.Sequential(base, v_out)
        
    def forward(self, state: torch.Tensor, actions: torch.Tensor):
        """
        Args:
            state: (batch_size, state_dim)
            actions: (batch_size, num_agents, action_dim)
        """
        state = state.flatten(1)
        actions = actions.flatten(1)
        x = torch.cat([state, actions], dim=-1)
        return torch.stack([critic(x) for critic in self.critics], dim=-1)


