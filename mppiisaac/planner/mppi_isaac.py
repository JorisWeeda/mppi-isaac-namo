from isaacgym import gymtorch

import copy
import torch

from typing import Callable, Optional

from control.mppi_isaac.mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper
from control.mppi_isaac.mppiisaac.utils.transport import bytes_to_torch, torch_to_bytes
from control.mppi_torch.mppi_torch.mppi import MPPIPlanner as MPPIPlanner

import control.mppi_isaac.mppiisaac as mppiisaac


torch.set_printoptions(precision=2, sci_mode=False)


class MPPIisaacPlanner(object):
    """
    Wrapper class that inherits from the MPPIPlanner and implements the required functions:
        dynamics, running_cost, and terminal_cost
    """

    def __init__(self, cfg, objective: Callable, prior: Optional[Callable] = None):
        self.cfg = cfg
        self.objective = objective
        self.done = False

        self.sim = IsaacGymWrapper(
            cfg["isaacgym"],
            actors=cfg["actors"],
            init_positions=cfg["initial_actor_positions"],
            num_envs=cfg["mppi"].num_samples,
            device=cfg["mppi"].device
        )

        if prior:
            self.prior = lambda state, t: prior.compute_command(self.sim)
        else:
            self.prior = None

        self.mppi = MPPIPlanner(
            cfg["mppi"],
            cfg["nx"],
            dynamics=self.dynamics,
            running_cost=self.running_cost,
            prior=self.prior,
        )
        # Note: place_holder variable to pass to mppi so it doesn't complain, while the real state is actually the isaacgym simulator itself.
        self.state_place_holder = torch.zeros((self.cfg["mppi"].num_samples, self.cfg["nx"]))
    
    def update_objective(self, waypoints):
        if isinstance(waypoints, bytes):
            waypoints = bytes_to_torch(waypoints, self.cfg["mppi"].device)

        self.objective.waypoints = waypoints

        # optimal_vel = (goal[:2] - init[:2]) / sum(abs(goal[:2] - init[:2]))
        # self.mppi.mean_action = torch.Tensor([*optimal_vel, 0.]) * self.mppi.u_max

    def dynamics(self, _, u, t=None):
        # Note: normally mppi passes the state as the first parameter in a dynamics call, but using isaacgym the state is already saved in the simulator itself, so we ignore it.
        # Note: t is an unused step dependent dynamics variable
        self.sim.apply_robot_cmd(u)
        self.sim.step()

        self.state_place_holder = self.sim.dof_state
        return self.state_place_holder, u

    def running_cost(self, _, u):
        # Note: again normally mppi passes the state as a parameter in the running cost call, but using isaacgym the state is already saved and accesible in the simulator itself, so we ignore it and pass a handle to the simulator.
        return self.objective.compute_cost(self.sim, u)

    def compute_action(self, q, qdot, obst=None, obst_tensor=None):
        self.objective.reset()
        self.sim.reset_robot_state(q, qdot)

        # NOTE: There are two different ways of updating obstacle root_states
        # Both update based on id in the list of obstacles
        if obst:
            self.sim.update_root_state_tensor_by_obstacles(obst)

        if obst_tensor:
            self.sim.update_root_state_tensor_by_obstacles_tensor(obst_tensor)

        u_cmd = self.mppi.command(self.state_place_holder)
        u = torch.zeros_like(u_cmd)
        u_idx = 0

        for actor in self.sim.env_cfg:
            if actor.type != "robot":
                continue

            dof_dict = self.sim._gym.get_actor_dof_dict(self.sim.envs[0], actor.handle)

            for _, i in dof_dict.items():
                u[i] = u_cmd[u_idx]
                u_idx += 1
        return u

    def reset_rollout_sim(self, dof_state_tensor, root_state_tensor, rigid_body_state_tensor):
        self.sim.visualize_link_buffer = []

        self.sim._dof_state[:] = bytes_to_torch(dof_state_tensor, self.cfg["mppi"].device)
        self.sim._root_state[:] = bytes_to_torch(root_state_tensor, self.cfg["mppi"].device)

        self.sim._gym.set_dof_state_tensor(
            self.sim._sim, gymtorch.unwrap_tensor(self.sim.dof_state)
        )
        self.sim._gym.set_actor_root_state_tensor(
            self.sim._sim, gymtorch.unwrap_tensor(self.sim.root_state)
        )

    def compute_action_tensor(self, dof_state_tensor, root_state_tensor, rigid_body_state_tensor=None):
        self.objective.reset()
        self.reset_rollout_sim(dof_state_tensor, root_state_tensor, rigid_body_state_tensor)
        return self.command()

    def command(self):
        return torch_to_bytes(self.mppi.command(self.state_place_holder))

    def add_to_env(self, env_cfg_additions):
        self.sim.add_to_envs(env_cfg_additions)

    def get_rollouts(self):
        # lines = lines[:, self.mppi.important_samples_indexes, :]
        # print(type(self.mppi.important_samples_indexes))
        if not self.sim._visualize_link_present:
            return torch_to_bytes(torch.zeros((1, 1, 1)))

        return torch_to_bytes(torch.stack(self.sim.visualize_link_buffer))

    def get_states(self):
        if self.mppi.states is None:
            return torch_to_bytes(torch.Tensor([]))

        return torch_to_bytes(self.mppi.states)

    def get_n_best_samples(self, n=1):
        states, _ = self.mppi.get_n_best_samples(n)
        return torch_to_bytes(states)

    def update_weights(self, weights):
        self.objective.weights = weights

    def update_mppi_params(self, params):
        self.cfg["mppi"].noise_sigma = params['noise_sigma']

        self.mppi = MPPIPlanner(
            self.cfg["mppi"],
            self.cfg["nx"],
            dynamics=self.dynamics,
            running_cost=self.running_cost,
            prior=self.prior,
        )