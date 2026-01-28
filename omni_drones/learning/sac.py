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
    传统控制器基座（PID/CTBR 等）的**接口壳**：占据与 RL actor 相同的“生态位”。

    约定：
    - 输入：obs 张量（与 RL actor 输入一致，即 `("agents","observation")`）
    - 输出：action 张量（与 RL actor 输出一致，即写入 `("agents","action")` 的那个 action，
      也就是 env transforms 之前的高层动作；shape/归一化/单位都要对齐）

    这里先提供架构/接口，具体控制律留给师弟实现。
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = int(action_dim)

        # TODO(you/teammate): 直接在代码里写死参数（不要 YAML）。
        # self.kp = ...
        # self.ki = ...
        # self.kd = ...

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (..., obs_dim)
        Returns:
            base_action: (..., action_dim)
        """
        # TODO(teammate): 在这里实现 pid_ctbr 控制律，输出需与 actor action 完全对齐。
        return torch.zeros((*obs.shape[:-1], self.action_dim), device=obs.device, dtype=obs.dtype)


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

        # ===== Base policy wiring (for BC loss) =====
        # base_policy 的输出会写入 ("info","base_action")，用于在 train_op 里做 bc_loss。
        # 注意：环境仍然执行 RL actor 写入的 ("agents","action")（除非你后续改策略）。
        self.base_policy = PIDCTBR(action_dim=self.action_dim).to(self.device)
        # 硬编码 bc loss 系数（不要 YAML）。需要启用时改成 >0。
        self.bc_coef = 0.0

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
        self.policy_out_keys = [self.act_name, f"{self.agent_spec.name}.logp"]
        
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
        # return tensordict.update({self.act_name: self.agent_spec.action_spec.zero()})
        actor_input = tensordict.select(*self.policy_in_keys)
        actor_input.batch_size = [*actor_input.batch_size, self.agent_spec.n]
        actor_output = self.actor(actor_input)
        # actor_output["action"].batch_size = tensordict.batch_size
        tensordict.update(actor_output)

        # 额外写入 base_action（不影响 env step），供 bc loss 使用
        with torch.no_grad():
            base_action = self.base_policy(actor_input[self.obs_name])
        tensordict.set(("info", "base_action"), base_action)
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

                    qs = self.critic(state, act)
                    q = torch.min(qs, dim=-1).values
                    actor_loss = (self.log_alpha.exp() * logp - q).mean()

                    # ===== BC loss scaffold =====
                    # 从 rollout 时写入的 ("info","base_action") 取出基座动作，计算 bc_loss
                    bc_loss = None
                    if ("info", "base_action") in transition.keys(True, True):
                        base_action = transition[("info", "base_action")]
                        bc_loss = F.mse_loss(act, base_action) #示意，不一定是这样

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
                        "actor_grad_norm": actor_grad_norm,
                        "entropy": -logp.mean(),
                        "alpha": self.log_alpha.exp().detach(),
                        "alpha_loss": alpha_loss,
                        **({"bc_loss": bc_loss.detach()} if bc_loss is not None else {}),
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

        return act, log_prob


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


