# SPDX-License-Identifier: Apache-2.0
"""MuJoCo implementation of the tensor interface used by Stage-Wise CMORL.

Loads the user's original task class and reward/stage/reset methods, replacing
only Isaac imports and the VecTask physics backend. This is a targeted adapter,
not an Isaac Gym emulator. The original source and assets are never rewritten.
"""
from __future__ import annotations

import ast
import copy
import re
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

TASKS = {"go1_backflip", "go1_sideflip", "go1_sideroll", "go1_twohand", "h1_backflip", "h1_twohand"}


def quat_rotate(q, v):
    # Isaac uses xyzw; MuJoCo uses wxyz. Tensors exposed here use xyzw.
    xyz, w = q[..., :3], q[..., 3:4]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_rotate_inverse(q, v):
    return quat_rotate(torch.cat((-q[..., :3], q[..., 3:4]), dim=-1), v)


torch_utils = SimpleNamespace(
    quat_rotate=quat_rotate, quat_rotate_inverse=quat_rotate_inverse,
    to_torch=lambda x, **kw: torch.tensor(x, **kw),
    torch_rand_float=lambda low, high, shape, device: low + (high - low) * torch.rand(shape, device=device),
)


def load_model(project: Path, robot: str):
    path = project / ("assets/go1/xml/go1.xml" if robot == "go1" else "assets/h1/mjcf/h1.xml")
    # Upstream Go1 XML has unquoted objtype=site and a stray '=' text node.
    # Normalize these syntax errors in memory, keeping source intact.
    xml = re.sub(r"<!--.*?-->", "", path.read_text(), flags=re.S)
    xml = xml.replace("objtype=site", 'objtype="site"')
    root = ET.fromstring(xml)
    for node in root.iter():
        if node.text and not node.text.strip(): node.text = None
        if node.tail and node.tail.strip() == "=": node.tail = None
    compiler = root.find("compiler")
    compiler.set("meshdir", str((path.parent / compiler.get("meshdir", "")).resolve()))
    compiler.set("fusestatic", "false")
    if robot == "go1":
        # The supplied MJCF combines foot collision geometry with calf bodies;
        # expose fixed foot bodies so the original contact costs stay distinct.
        for body in root.findall(".//body"):
            if body.get("name", "").endswith("_calf"):
                foot = ET.SubElement(body, "body", name=body.get("name").replace("calf", "foot"))
                for geom in list(body.findall("geom")):
                    pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
                    if len(pos) == 3 and pos[2] < -0.2:
                        body.remove(geom)
                        foot.append(geom)
    if not root.findall(".//geom[@type='plane']"):
        world = root.find("worldbody")
        ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 1", rgba="0.18 0.22 0.28 1")
        ET.SubElement(world, "light", pos="0 0 4", dir="0 0 -1")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode")), path


class TensorPhysics:
    """Real MuJoCo stepping and explicitly ordered CPU tensors."""
    def __init__(self, owner):
        self.owner = owner
        n, j, b = owner.num_envs, owner.num_dofs, owner.num_bodies
        self.root = torch.zeros(n, 13)
        self.dofs = torch.zeros(n, j, 2)
        self.forces = torch.zeros(n, j)
        self.contacts = torch.zeros(n, b, 3)
        self.bodies = torch.zeros(n, b, 13)
        self.torques = np.zeros((n, j))
        self.position_targets = None

    def acquire_actor_root_state_tensor(self, sim): return self.root
    def acquire_dof_state_tensor(self, sim): return self.dofs
    def acquire_dof_force_tensor(self, sim): return self.forces
    def acquire_net_contact_force_tensor(self, sim): return self.contacts
    def acquire_rigid_body_state_tensor(self, sim): return self.bodies

    def refresh_dof_state_tensor(self, sim):
        o = self.owner
        for i, data in enumerate(o.datas):
            self.dofs[i, :, 0] = torch.from_numpy(data.qpos[o.qpos_ids].astype(np.float32))
            self.dofs[i, :, 1] = torch.from_numpy(data.qvel[o.qvel_ids].astype(np.float32))

    def refresh_dof_force_tensor(self, sim):
        for i, data in enumerate(self.owner.datas):
            self.forces[i] = torch.from_numpy(data.qfrc_actuator[self.owner.qvel_ids].astype(np.float32))

    def refresh_actor_root_state_tensor(self, sim):
        o = self.owner
        for i, data in enumerate(o.datas):
            self.root[i, :3] = torch.from_numpy(data.qpos[:3].astype(np.float32))
            self.root[i, 3:7] = torch.from_numpy(data.qpos[[4, 5, 6, 3]].astype(np.float32))
            self.root[i, 7:10] = torch.from_numpy(data.qvel[:3].astype(np.float32))
            self.root[i, 10:] = torch.from_numpy((data.xmat[1].reshape(3, 3) @ data.qvel[3:6]).astype(np.float32))

    def refresh_net_contact_force_tensor(self, sim):
        o = self.owner
        values = self.contacts.numpy()
        values.fill(0)
        wrench = np.zeros(6)
        for i, data in enumerate(o.datas):
            for index in range(data.ncon):
                contact = data.contact[index]
                mujoco.mj_contactForce(o.model, data, index, wrench)
                force = contact.frame.reshape(3, 3).T @ wrench[:3]
                b1, b2 = o.model.geom_bodyid[contact.geom1], o.model.geom_bodyid[contact.geom2]
                values[i, b1] -= force
                values[i, b2] += force

    def refresh_rigid_body_state_tensor(self, sim):
        o = self.owner
        velocity = np.zeros(6)
        values = self.bodies.numpy()
        for i, data in enumerate(o.datas):
            values[i, :, :3] = data.xpos
            values[i, :, 3:7] = data.xquat[:, [1, 2, 3, 0]]
            for b in range(o.num_bodies):
                mujoco.mj_objectVelocity(o.model, data, mujoco.mjtObj.mjOBJ_BODY, b, velocity, 0)
                values[i, b, 7:10], values[i, b, 10:13] = velocity[3:], velocity[:3]

    def set_dof_state_tensor_indexed(self, sim, states, ids, count):
        o = self.owner
        for index in ids.tolist():
            data = o.datas[index]
            data.qpos[o.qpos_ids] = states[index, :, 0].numpy()
            data.qvel[o.qvel_ids] = states[index, :, 1].numpy()

    def set_actor_root_state_tensor_indexed(self, sim, states, ids, count):
        o = self.owner
        for index in ids.tolist():
            data = o.datas[index]
            state = states[index].numpy()
            data.qpos[:3], data.qpos[3:7] = state[:3], state[[6, 3, 4, 5]]
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
            data.qvel[:3], data.qvel[3:6] = state[7:10], rotation.reshape(3, 3).T @ state[10:13]
            data.qacc_warmstart[:] = 0
            data.ctrl[:] = 0
            mujoco.mj_forward(o.model, data)

    def set_dof_actuation_force_tensor(self, sim, torques):
        self.torques[:] = torques.detach().cpu().numpy()

    def set_dof_position_target_tensor(self, sim, targets):
        self.position_targets = targets.detach().cpu().numpy().copy()

    def simulate(self, sim):
        o = self.owner
        for i, data in enumerate(o.datas):
            torque = self.torques[i]
            if self.position_targets is not None:
                torque = o.kp * (self.position_targets[i] - data.qpos[o.qpos_ids]) - o.kd * data.qvel[o.qvel_ids]
            torque = np.clip(torque, -o.effort_limits, o.effort_limits)
            data.ctrl[o.actuator_ids] = torque
            mujoco.mj_step(o.model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError("MuJoCo produced non-finite state")
        # mj_forward synchronizes positions/contact solve with the integrated
        # state; task refresh calls must not observe pre-integration transforms.
        for data in o.datas:
            mujoco.mj_forward(o.model, data)

    def fetch_results(self, sim, wait): pass  # CPU mj_step already completed.

    def find_actor_rigid_body_handle(self, env, actor, name):
        aliases = {"base": "trunk", "torso": "torso_link"}
        name = aliases.get(name, name)
        index = mujoco.mj_name2id(self.owner.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if index < 0: raise ValueError(f"MuJoCo model is missing task body {name}")
        return index


class MujocoVecTask:
    def __init__(self, config, **kwargs):
        self.cfg = config
        env = config["env"]
        if env.get("randomize", {}).get("is_randomized", False):
            raise ValueError("Stage-Wise MuJoCo adapter currently requires env.randomize.is_randomized=false; PhysX material randomization is not equivalent.")
        self.device = self.rl_device = "cpu"
        self.num_envs, self.num_obs, self.num_states = int(env["numEnvs"]), int(env["numObservations"]), int(env["numStates"])
        self.num_actions = self.num_acts = int(env["numActions"])
        self.control_freq_inv = int(env["controlFrequencyInv"])
        # Match IsaacGymEnvs VecTask defaults. Stage-Wise uses an unbounded
        # Gaussian actor; clipping at 1 silently prevents the intended joint travel.
        self.clip_actions, self.clip_obs = float(env.get("clipActions", np.inf)), float(env.get("clipObservations", np.inf))
        self.viewer, self.force_render, self.sim = None, False, None
        root = Path(config["roboverse_project"])
        robot = config["name"][:2].lower() if config["name"].startswith("H1") else "go1"
        self.model, self.model_path = load_model(root, robot)
        self.model.opt.timestep = float(config["sim"]["sim_dt"])
        self.model.opt.gravity[:] = config["sim"]["gravity"]
        # Derive ordering and limits from the project's URDF, never dictionary order.
        urdf = ET.parse(root / "assets" / env["urdf_asset"]["file"])
        joints = [j for j in urdf.findall("joint") if j.get("type") in {"revolute", "continuous", "prismatic"}]
        self.dof_names = [j.get("name") for j in joints]
        self.num_dofs = len(joints)
        if self.num_dofs != self.num_actions: raise ValueError("task/URDF action dimensions disagree")
        ids = np.array([self.model.joint(name).id for name in self.dof_names])
        self.qpos_ids, self.qvel_ids = self.model.jnt_qposadr[ids], self.model.jnt_dofadr[ids]
        self.actuator_ids = np.array([int(np.flatnonzero(self.model.actuator_trnid[:, 0] == j)[0]) for j in ids])
        self.effort_limits = np.minimum(
            [float(j.find("limit").get("effort")) for j in joints],
            np.abs(self.model.actuator_ctrlrange[self.actuator_ids]).max(axis=1))
        self.default_dof_pos_lower_limits = torch.tensor(self.model.jnt_range[ids, 0], dtype=torch.float32)
        self.default_dof_pos_upper_limits = torch.tensor(self.model.jnt_range[ids, 1], dtype=torch.float32)
        self.default_dof_vel_upper_limits = torch.tensor([float(j.find("limit").get("velocity")) for j in joints])
        self.default_dof_torques_upper_limits = torch.tensor(self.effort_limits, dtype=torch.float32)
        def gains(key):
            value = env["control"][key]
            return np.array([next((v for k, v in value.items() if k in n), 0) for n in self.dof_names]) if isinstance(value, dict) else float(value)
        self.kp, self.kd = gains("stiffness"), gains("damping")
        self.num_bodies = self.model.nbody
        self.link_names = [self.model.body(i).name for i in range(1, self.num_bodies)]
        self.datas = [mujoco.MjData(self.model) for _ in range(self.num_envs)]
        for data in self.datas: mujoco.mj_forward(self.model, data)
        self.gym = TensorPhysics(self)
        self.env_handles, self.robot_handles = [0], [0]
        self.friction_coeffs = torch.ones(self.num_envs, 1)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1)
        self.model.geom_friction[:, 0] = 1.0
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs)
        self.states_buf = torch.zeros(self.num_envs, self.num_states)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.reset_buf = torch.zeros_like(self.progress_buf)
        self.extras, self.obs_dict = {}, {}
        self._setup_indices()

    def reset(self):
        return {"obs": self.obs_buf.clone(), "states": self.states_buf.clone()}

    def get_state(self): return self.states_buf.clone()

    def step(self, actions):
        self.pre_physics_step(actions.clamp(-self.clip_actions, self.clip_actions))
        for _ in range(self.control_freq_inv): self.gym.simulate(self.sim)
        self.post_physics_step()
        return {"obs": self.obs_buf.clone(), "states": self.states_buf.clone()}, self.rew_buf.clone(), self.extras["dones"].clone(), self.extras


def create_env(project, task_file, cfg, num_envs=8, history_len=3):
    path = Path(task_file).resolve()
    if path.stem not in TASKS: raise ValueError(f"No Stage-Wise task adapter for {path.stem}")
    tree = ast.parse(path.read_text(), filename=str(path))
    # Import real task Python but provide our own explicit physics API. Keep
    # original definitions, including reward code and user edits, in memory.
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and
                 (node.module or "").startswith(("isaacgym", "isaacgymenvs")))]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Env")
    create = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "create_sim")
    index = next(i for i, node in enumerate(create.body) if isinstance(node, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == "base_names" for t in node.targets))
    create.name, create.body = "_setup_indices", create.body[index:]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef): node.decorator_list = []
    namespace = {"__file__": str(path), "__name__": "roboverse_stagewise_task", "VecTask": MujocoVecTask,
                 "torch_utils": torch_utils, "gymtorch": SimpleNamespace(wrap_tensor=lambda x: x, unwrap_tensor=lambda x: x)}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    config = copy.deepcopy(cfg)
    config["roboverse_project"] = str(Path(project).resolve())
    config["env"]["num_envs"], config["env"]["history_len"] = num_envs, history_len
    return namespace["Env"](cfg=config, rl_device="cpu", sim_device="cpu", graphics_device_id=-1,
                            headless=True, virtual_screen_capture=False, force_render=False)
