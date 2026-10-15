"""Editable single-world MuJoCo workbench. Physics and rendering share one owner thread."""
from __future__ import annotations

import csv
import copy
import io
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image

from studio_project import import_spec, read_archive, save_archive

STATE = mujoco.mjtState.mjSTATE_INTEGRATION
WORLD_XML = '''<mujoco model="Untitled world">
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual><global offwidth="1920" offheight="1200"/><headlight ambient="0.35 0.35 0.4" diffuse="0.7 0.7 0.7"/><rgba haze="0.08 0.1 0.14 1"/></visual>
  <asset><texture name="grid" type="2d" builtin="checker" rgb1="0.16 0.19 0.23" rgb2="0.21 0.25 0.29" width="512" height="512"/><material name="floor" texture="grid" texrepeat="12 12" texuniform="true" reflectance="0.08"/></asset>
  <worldbody><light pos="0 -3 6" dir="0 0 -1" diffuse="0.8 0.85 0.95"/>
    <geom name="ground" type="plane" size="10 10 0.1" material="floor" friction="0.8 0.005 0.0001"/>
    <body name="crate" pos="0 0 0.7"><freejoint name="crate_free"/><geom name="crate_shape" type="box" size="0.2 0.2 0.2" mass="1" rgba="0.2 0.65 0.85 1"/></body>
  </worldbody></mujoco>'''


def number(value, low=-1e6, high=1e6):
    value = float(value)
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"数值必须在 {low:g} 与 {high:g} 之间")
    return value


def vector(value, n=3):
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"需要 {n} 个数值")
    return [number(x) for x in value]


class World:
    def __init__(self):
        self.spec = self.model = self.data = self.renderer = None
        self.width, self.height = 960, 640
        self.camera = mujoco.MjvCamera()
        self.options = mujoco.MjvOption()
        self.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        self.selected = 0
        self.running = False
        self.speed = 1.0
        self.project_path = ""
        self.dirty = False
        self.revision = 0
        self.undo_stack, self.redo_stack = [], []
        self.controllers = {}
        self.trajectory = None
        self.recording = False
        self.record_states, self.record_times = [], []
        self.record_spec = None
        self.replay_states = self.replay_times = None
        self.replay_index = 0
        self._record_next = 0
        self.record_limit = 30000
        self.force_until = 0
        self.actual_rtf = 0.0
        self.last_notice = ""
        self.install(mujoco.MjSpec.from_string(WORLD_XML), reset_camera=True)

    def capture_state(self):
        values = np.empty(mujoco.mj_stateSize(self.model, STATE))
        mujoco.mj_getState(self.model, self.data, values, STATE)
        return values

    def restore_state(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (mujoco.mj_stateSize(self.model, STATE),) or not np.isfinite(values).all():
            raise ValueError("物理状态与模型不匹配")
        mujoco.mj_setState(self.model, self.data, values, STATE)
        mujoco.mj_forward(self.model, self.data)

    def snapshot(self):
        return (self.spec.copy(), self.capture_state().copy(), {k: dict(v) for k, v in self.controllers.items()}, copy.deepcopy(self.trajectory))

    @staticmethod
    def ensure_names(spec):
        # Stable names make state transfer, controllers and the scene inspector unambiguous.
        for prefix, elements in (("body", spec.bodies), ("joint", spec.joints), ("geom", spec.geoms),
                                 ("actuator", spec.actuators), ("sensor", spec.sensors)):
            used = {element.name for element in elements if element.name}
            for index, element in enumerate(elements):
                if not element.name:
                    name = f"studio_{prefix}_{index}"
                    while name in used: name += "_"
                    element.name = name
                    used.add(name)

    def install(self, spec, preserve=False, reset_camera=False):
        self.ensure_names(spec)
        model = spec.compile()
        model.vis.global_.offwidth = max(1920, self.width)
        model.vis.global_.offheight = max(1200, self.height)
        data = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        if preserve and self.model is not None:
            data.time = self.data.time
            for j in range(model.njnt):
                name = model.joint(j).name
                old = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) if name else -1
                if old >= 0 and model.jnt_type[j] == self.model.jnt_type[old]:
                    nq = 7 if model.jnt_type[j] == 0 else 4 if model.jnt_type[j] == 1 else 1
                    nv = 6 if model.jnt_type[j] == 0 else 3 if model.jnt_type[j] == 1 else 1
                    q, oq = model.jnt_qposadr[j], self.model.jnt_qposadr[old]
                    v, ov = model.jnt_dofadr[j], self.model.jnt_dofadr[old]
                    data.qpos[q:q+nq] = self.data.qpos[oq:oq+nq]
                    data.qvel[v:v+nv] = self.data.qvel[ov:ov+nv]
            for a in range(model.nu):
                old = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.actuator(a).name)
                if old >= 0:
                    data.ctrl[a] = self.data.ctrl[old]
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, self.height, self.width)
        previous = self.renderer
        self.spec, self.model, self.data, self.renderer = spec, model, data, renderer
        if previous:
            previous.close()
        self.selected = min(self.selected, model.nbody - 1)
        self.controllers = {k: v for k, v in self.controllers.items()
                            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, k) >= 0}
        if self.trajectory and any(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) < 0 for name in self.trajectory["values"]):
            self.trajectory = None
        if reset_camera:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.lookat[:] = model.stat.center
            self.camera.distance = max(2.5, 2.5 * model.stat.extent)
            self.camera.azimuth, self.camera.elevation = 135, -25
        self.revision += 1

    def require_editable(self):
        if self.running or self.recording or self.replay_states is not None:
            raise ValueError("请先暂停仿真并结束录制／回放，再编辑场景")

    def edit(self, change):
        self.require_editable()
        before = self.snapshot()
        candidate = self.spec.copy()
        change(candidate)
        self.install(candidate, preserve=True)
        self.undo_stack.append(before)
        self.undo_stack = self.undo_stack[-12:]
        self.redo_stack.clear()
        self.dirty = True

    def undo(self, redo=False):
        self.require_editable()
        source, dest = (self.redo_stack, self.undo_stack) if redo else (self.undo_stack, self.redo_stack)
        if not source:
            return
        current = self.snapshot()
        spec, state, controls, trajectory = source[-1]
        self.install(spec.copy())
        self.restore_state(state)
        self.controllers = controls
        self.trajectory = trajectory
        source.pop()
        dest.append(current)
        self.dirty = True

    def load(self, path=None, builtin=None):
        self.require_editable()
        if path:
            candidate = import_spec(path)
        elif builtin == "g1":
            candidate = import_spec(str(Path(__file__).resolve().parent.parent / "models/unitree_g1/scene.xml"))
        elif builtin and builtin != "world":
            if builtin not in {"cartpole", "acrobot"}:
                raise ValueError(f"Unknown built-in model: {builtin}")
            candidate = import_spec(str(Path(__file__).resolve().parent.parent / "models" / f"{builtin}.xml"))
        else:
            candidate = mujoco.MjSpec.from_string(WORLD_XML)
        self.install(candidate, reset_camera=True)
        self.controllers.clear()
        self.trajectory = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.project_path = ""
        self.selected = 0
        self.dirty = True
        self.record_states, self.record_times = [], []
        self.record_spec = None

    def insert_model(self, cmd):
        self.require_editable()
        path = cmd.get("path")
        if cmd.get("builtin") == "g1":
            path = str(Path(__file__).resolve().parent.parent / "models/unitree_g1/g1.xml")
        if not path:
            raise ValueError("请选择模型文件")
        robot = import_spec(path)
        self.ensure_names(robot)
        robot_model = robot.compile()
        robot_data = mujoco.MjData(robot_model)
        if robot_model.nkey: mujoco.mj_resetDataKeyframe(robot_model, robot_data, 0)
        position = vector(cmd.get("position", [1, 0, 0]))
        # Attach transforms the model frame, but keyframe free-joint translations stay absolute.
        for key in robot.keys:
            key_qpos = list(key.qpos)
            for j in range(robot_model.njnt):
                if robot_model.jnt_type[j] == 0 and len(key_qpos) >= robot_model.jnt_qposadr[j]+3:
                    q = int(robot_model.jnt_qposadr[j])
                    for axis in range(3): key_qpos[q+axis] += position[axis]
            key.qpos = key_qpos
        robot.nkey = 0  # keys themselves determine storage; avoid conflicting allocation hints
        prefix = f"robot_{self.revision}/"
        def change(spec):
            spec.copy_during_attach = True
            frame = spec.worldbody.add_frame(name=prefix+"origin", pos=position)
            spec.attach(robot, frame=frame, prefix=prefix)
        self.edit(change)
        # Keyframe free-joint coordinates are absolute: explicitly transform the inserted reset pose.
        for j in range(robot_model.njnt):
            destination = self.model.joint(prefix+robot_model.joint(j).name).id
            q, source_q = self.model.jnt_qposadr[destination], robot_model.jnt_qposadr[j]
            nq = 7 if robot_model.jnt_type[j] == 0 else 4 if robot_model.jnt_type[j] == 1 else 1
            self.data.qpos[q:q+nq] = robot_data.qpos[source_q:source_q+nq]
            if robot_model.jnt_type[j] == 0:
                self.data.qpos[q:q+3] += position
        for a in range(robot_model.nu):
            self.data.ctrl[self.model.actuator(prefix+robot_model.actuator(a).name).id] = robot_data.ctrl[a]
        mujoco.mj_forward(self.model, self.data)
        for body in self.spec.bodies:
            if body.name.startswith(prefix):
                self.selected = self.model.body(body.name).id
                break

    def add(self, cmd):
        shape = cmd.get("shape", "box")
        kinds = {"box": mujoco.mjtGeom.mjGEOM_BOX, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
                 "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER}
        if shape not in kinds:
            raise ValueError("不支持的几何体")
        index = 1
        name = f"{shape}_{index}"
        while self.spec.body(name):
            index += 1
            name = f"{shape}_{index}"
        position = vector(cmd.get("position", [0.5, 0, 1]))
        size = [number(x, 0.005, 100) for x in vector(cmd.get("size", [.15, .15, .15]))]
        mass = number(cmd.get("mass", 1), .001, 100000)
        def change(spec):
            body = spec.worldbody.add_body(name=name, pos=position)
            if cmd.get("dynamic", True):
                body.add_freejoint(name=name + "_free")
            body.add_geom(name=name + "_shape", type=kinds[shape], size=size, mass=mass,
                          rgba=[.3, .7, .85, 1], friction=[.8, .005, .0001])
        self.edit(change)
        self.selected = self.model.body(name).id

    def remove_selected(self):
        if self.selected == 0:
            raise ValueError("不能删除世界根节点")
        name = self.model.body(self.selected).name
        self.edit(lambda spec: spec.delete(spec.body(name)))
        self.selected = 0

    def add_imu(self, cmd):
        body_id = int(cmd.get("id", self.selected))
        if not 0 < body_id < self.model.nbody:
            raise ValueError("请选择一个物体")
        name = self.model.body(body_id).name
        site_name = name + "_imu"
        if self.spec.site(site_name):
            raise ValueError("该物体已有工作台 IMU")
        def change(spec):
            body = spec.body(name)
            body.add_site(name=site_name, size=[.008, .008, .008], rgba=[.2, .9, .5, 1])
            spec.add_sensor(name=name+"_gyro", type=mujoco.mjtSensor.mjSENS_GYRO,
                            objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site_name)
            spec.add_sensor(name=name+"_accelerometer", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                            objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site_name)
        self.edit(change)

    def body_edit(self, cmd):
        body_id = int(cmd.get("id", self.selected))
        if not 0 < body_id < self.model.nbody:
            raise ValueError("请选择一个物体")
        name = self.model.body(body_id).name
        position = vector(cmd["position"]) if "position" in cmd else None
        quaternion = None
        if "quaternion" in cmd:
            quaternion = np.asarray(vector(cmd["quaternion"], 4))
            length = np.linalg.norm(quaternion)
            if length < 1e-8:
                raise ValueError("旋转四元数不能为零")
            quaternion /= length
        mass = number(cmd["mass"], .0001, 1e6) if "mass" in cmd else None
        friction = number(cmd["friction"], 0, 100) if "friction" in cmd else None
        def change(spec):
            body = spec.body(name)
            if position is not None:
                body.pos = position
            if quaternion is not None:
                body.quat = quaternion
            if mass is not None:
                oldmass = self.model.body_mass[body_id]
                if oldmass <= 0:
                    raise ValueError("该节点没有可编辑的惯性")
                body.explicitinertial = True
                body.mass = mass
                body.ipos = self.model.body_ipos[body_id]
                body.iquat = self.model.body_iquat[body_id]
                body.inertia = self.model.body_inertia[body_id] * mass / oldmass
            if friction is not None:
                for geom in body.geoms:
                    geom.friction[0] = friction
        self.edit(change)
        # Floating bodies live in qpos; changing their reference transform alone would not move them.
        if position is not None or quaternion is not None:
            b = self.model.body(name).id
            j = self.model.body_jntadr[b]
            if j >= 0 and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                q = self.model.jnt_qposadr[j]
                if position is not None: self.data.qpos[q:q+3] = position
                if quaternion is not None: self.data.qpos[q+3:q+7] = quaternion
                v = self.model.jnt_dofadr[j]
                self.data.qvel[v:v+6] = 0
                mujoco.mj_forward(self.model, self.data)

    def physics(self, cmd):
        gravity = vector(cmd.get("gravity", self.model.opt.gravity.tolist()))
        dt = number(cmd.get("timestep", self.model.opt.timestep), .00001, .05)
        iterations = int(number(cmd.get("iterations", self.model.opt.iterations), 1, 1000))
        def change(spec):
            spec.option.gravity = gravity
            spec.option.timestep = dt
            spec.option.iterations = iterations
        self.edit(change)

    def set_joint(self, cmd):
        self.require_editable()
        j = int(cmd["id"])
        if not 0 <= j < self.model.njnt or self.model.jnt_type[j] not in (2, 3):
            raise ValueError("只支持标量关节位置编辑")
        value = number(cmd["value"])
        if self.model.jnt_limited[j]:
            value = float(np.clip(value, *self.model.jnt_range[j]))
        self.undo_stack.append(self.snapshot())
        self.undo_stack = self.undo_stack[-12:]
        self.redo_stack.clear()
        self.data.qpos[self.model.jnt_qposadr[j]] = value
        self.data.qvel[self.model.jnt_dofadr[j]] = 0
        mujoco.mj_forward(self.model, self.data)
        self.dirty = True

    def control(self, cmd):
        if self.replay_states is not None:
            raise ValueError("请先退出回放")
        if self.trajectory is not None:
            raise ValueError("请先停止轨迹控制")
        a = int(cmd["id"])
        if not 0 <= a < self.model.nu:
            raise ValueError("执行器不存在")
        mode = cmd.get("mode", "direct")
        if mode not in ("direct", "pd", "off"):
            raise ValueError("未知控制模式")
        if mode == "pd" and not self.pd_compatible(a):
            raise ValueError("PD 仅适用于连接标量关节的无偏置电机；位置伺服请使用原生输入")
        self.controllers[self.model.actuator(a).name] = {
            "mode": mode, "target": number(cmd.get("target", 0)),
            "kp": number(cmd.get("kp", 30), 0, 10000), "kd": number(cmd.get("kd", 3), 0, 1000)}
        self.dirty = True

    def pd_compatible(self, a):
        j = self.model.actuator_trnid[a, 0]
        return (self.model.actuator_trntype[a] == 0 and j >= 0 and self.model.jnt_type[j] in (2, 3)
                and self.model.actuator_biastype[a] == 0 and self.model.actuator_dyntype[a] == 0
                and self.model.actuator_gaintype[a] == 0
                and abs(self.model.actuator_gear[a, 0] * self.model.actuator_gainprm[a, 0]) > 1e-12)

    def apply_controls(self):
        if self.trajectory is not None:
            trajectory = self.trajectory
            elapsed = max(0, self.data.time-trajectory["start"])
            for name, values in trajectory["values"].items():
                a = self.model.actuator(name).id
                value = np.interp(elapsed, trajectory["times"], values)
                if self.model.actuator_ctrllimited[a]: value = np.clip(value, *self.model.actuator_ctrlrange[a])
                self.data.ctrl[a] = value
            return
        for name, control in self.controllers.items():
            a = self.model.actuator(name).id
            target = control["target"]
            if control["mode"] == "off":
                target = 0
            elif control["mode"] == "pd":
                j = self.model.actuator_trnid[a, 0]
                q, v = self.model.jnt_qposadr[j], self.model.jnt_dofadr[j]
                torque = control["kp"] * (target - self.data.qpos[q]) - control["kd"] * self.data.qvel[v]
                target = torque / (self.model.actuator_gear[a, 0] * self.model.actuator_gainprm[a, 0])
            if self.model.actuator_ctrllimited[a]:
                target = np.clip(target, *self.model.actuator_ctrlrange[a])
            self.data.ctrl[a] = target

    def load_trajectory(self, path):
        if self.replay_states is not None:
            raise ValueError("请先退出回放")
        source = Path(path)
        if source.stat().st_size > 10*1024*1024:
            raise ValueError("轨迹文件不能超过 10 MB")
        with source.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            fields = reader.fieldnames or []
        if "time_s" not in fields or not 2 <= len(rows) <= 100000:
            raise ValueError("CSV 需要 time_s 列和至少两行数据（最多 100000 行）")
        times = [number(row["time_s"], 0, 86400) for row in rows]
        if times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError("轨迹时间必须从 0 开始且严格递增")
        values = {}
        for a in range(self.model.nu):
            name = self.model.actuator(a).name
            column = "ctrl_"+name
            if column in fields:
                values[name] = [number(row[column]) for row in rows]
        if not values:
            raise ValueError("没有匹配执行器的 ctrl_<执行器名> 列")
        self.trajectory = {"name": source.name, "times": times, "values": values, "start": float(self.data.time)}
        self.dirty = True

    def step(self, count=1):
        if self.replay_states is not None:
            self.seek(min(len(self.replay_times)-1, self.replay_index + count))
            return
        for _ in range(count):
            self.apply_controls()
            if self.data.time >= self.force_until:
                self.data.xfrc_applied[:] = 0
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                self.running = False
                raise ValueError("物理状态发散，请减小步长并检查质量、约束和控制器")
            if self.recording and self.data.time + 1e-9 >= self._record_next:
                self.record_states.append(self.capture_state())
                self.record_times.append(float(self.data.time))
                self._record_next += .02
                if self._record_next <= self.data.time:
                    self._record_next = self.data.time + .02
                if len(self.record_states) >= self.record_limit:
                    self.recording = False
                    self.last_notice = "录制达到内存上限，已自动停止；请导出录制"
        mujoco.mj_forward(self.model, self.data)
        self.dirty = True

    def reset(self, key=-1):
        if self.recording or self.replay_states is not None:
            raise ValueError("请先结束录制／回放")
        self.running = False
        if 0 <= key < self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.force_until = 0
        self.controllers.clear()
        self.trajectory = None
        mujoco.mj_forward(self.model, self.data)
        self.dirty = True

    def force(self, cmd):
        if self.replay_states is not None:
            raise ValueError("回放中不能施加外力")
        body = int(cmd.get("id", self.selected))
        if not 0 < body < self.model.nbody:
            raise ValueError("请选择一个物体")
        self.data.xfrc_applied[body, :3] = vector(cmd.get("force", [10, 0, 0]))
        self.force_until = self.data.time + number(cmd.get("duration", .2), .001, 10)

    def start_recording(self):
        if self.replay_states is not None:
            raise ValueError("请先退出回放")
        self.record_spec = self.spec.copy()
        self.record_states, self.record_times = [self.capture_state()], [float(self.data.time)]
        self.record_limit = min(30000, max(100, 128 * 1024 * 1024 // (self.record_states[0].nbytes + 64)))
        self._record_next = self.data.time + .02
        self.recording = True

    def save(self, path):
        save_archive(path, self.spec, {"kind": "project", "state": self.capture_state().tolist(),
                     "controllers": self.controllers, "speed": self.speed,
                     "trajectory": self.trajectory,
                     "camera": self.camera_info(), "force_remaining": max(0, self.force_until-self.data.time)})
        self.project_path = path
        self.dirty = False

    def open(self, path, recording=False):
        self.require_editable()
        spec, metadata, states, times = read_archive(path)
        model = spec.compile()
        expected = mujoco.mj_stateSize(model, STATE)
        state = np.asarray(metadata.get("state", []))
        controls = metadata.get("controllers", {})
        trajectory = metadata.get("trajectory")
        if trajectory:
            times_array = np.asarray(trajectory["times"], dtype=float)
            if times_array.ndim != 1 or len(times_array) < 2 or len(times_array) > 100000 or not np.isfinite(times_array).all() or times_array[0] != 0 or np.any(np.diff(times_array) <= 0):
                raise ValueError("项目轨迹时间无效")
            number(trajectory["start"], 0, 1e12)
            for name, values in trajectory["values"].items():
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) < 0 or len(values) != len(times_array) or not np.isfinite(np.asarray(values, dtype=float)).all():
                    raise ValueError("项目轨迹数据无效")
        speed = number(metadata.get("speed", 1), .1, 10)
        remaining = number(metadata.get("force_remaining", 0), 0, 10)
        camera = metadata.get("camera")
        if camera:
            camera = {"lookat": vector(camera["lookat"]),
                      "azimuth": number(camera["azimuth"]), "elevation": number(camera["elevation"], -89, 89),
                      "distance": number(camera["distance"], .05, 1000)}
        if recording:
            if states is None or states.shape[1] != expected:
                raise ValueError("录制与模型状态不匹配")
        elif state.shape != (expected,) or not np.isfinite(state).all():
            raise ValueError("项目状态无效")
        if not isinstance(controls, dict):
            raise ValueError("控制器配置无效")
        for name, c in controls.items():
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) < 0 or c.get("mode") not in ("off", "pd", "direct"):
                raise ValueError("控制器配置与模型不匹配")
            for field in ("target", "kp", "kd"):
                number(c[field])
        self.install(spec, reset_camera=True)
        self.controllers = controls
        self.trajectory = trajectory
        self.running = False
        self.selected = 0
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.record_states, self.record_times, self.record_spec = [], [], None
        self.speed = speed
        if camera:
            self.camera.lookat[:] = camera["lookat"]
            self.camera.azimuth, self.camera.elevation, self.camera.distance = camera["azimuth"], camera["elevation"], camera["distance"]
        if recording:
            self.replay_states, self.replay_times = states, times
            self.seek(0)
            self.project_path = ""
        else:
            self.restore_state(state)
            self.force_until = self.data.time + remaining
            self.project_path = path
        self.dirty = False

    def replay(self):
        if self.recording:
            raise ValueError("请先停止录制")
        if not self.record_states:
            raise ValueError("没有录制数据")
        self.running = False
        self.install(self.record_spec.copy())
        self.replay_states, self.replay_times = np.asarray(self.record_states), np.asarray(self.record_times)
        self.seek(0)

    def seek(self, index):
        if self.replay_states is None:
            raise ValueError("没有正在回放的录制")
        self.replay_index = int(np.clip(index, 0, len(self.replay_times)-1))
        self.restore_state(self.replay_states[self.replay_index])

    def export_recording(self, path):
        if self.recording:
            raise ValueError("请先停止录制")
        if not self.record_states and self.replay_states is None:
            raise ValueError("没有录制数据")
        spec = self.record_spec if self.record_states else self.spec
        states = self.record_states if self.record_states else self.replay_states
        times = self.record_times if self.record_times else self.replay_times
        save_archive(path, spec, {"kind": "recording", "controllers": {}}, states, times)

    def export_csv(self, path):
        if self.recording:
            raise ValueError("请先停止录制")
        states = self.record_states if self.record_states else self.replay_states
        if states is None or len(states) == 0:
            raise ValueError("没有录制数据")
        spec = self.record_spec if self.record_states else self.spec
        model = spec.compile()
        data = mujoco.MjData(model)
        # Export from a separate MjData: viewing or exporting never changes the live world.
        columns = ["time_s"] + [f"qpos_{i}" for i in range(model.nq)] + [f"qvel_{i}" for i in range(model.nv)]
        columns += [f"ctrl_{model.actuator(i).name or i}" for i in range(model.nu)]
        columns += [f"sensor_{i}" for i in range(model.nsensordata)] + ["contacts"]
        destination = Path(path)
        import tempfile, os
        fd, temporary = tempfile.mkstemp(prefix=".mjstudio-csv-", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(columns)
                for state in states:
                    mujoco.mj_setState(model, data, np.asarray(state), STATE)
                    mujoco.mj_forward(model, data)
                    writer.writerow([data.time, *data.qpos, *data.qvel, *data.ctrl, *data.sensordata, data.ncon])
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)

    def camera_info(self):
        return {"azimuth": float(self.camera.azimuth), "elevation": float(self.camera.elevation),
                "distance": float(self.camera.distance), "lookat": self.camera.lookat.tolist()}

    def camera_command(self, cmd):
        if cmd.get("focus"):
            self.camera.lookat[:] = self.data.xpos[self.selected] if self.selected else self.model.stat.center
        if "orbit" in cmd:
            delta = vector(cmd["orbit"], 2)
            self.camera.azimuth += delta[0]
            self.camera.elevation = np.clip(self.camera.elevation+delta[1], -89, 89)
        if "zoom" in cmd:
            self.camera.distance = np.clip(self.camera.distance * number(cmd["zoom"], .1, 10), .05, 1000)
        if "pan" in cmd:
            dx, dy = vector(cmd["pan"], 2)
            angle = np.deg2rad(self.camera.azimuth)
            self.camera.lookat += self.camera.distance * np.array([dx*np.sin(angle), -dx*np.cos(angle), dy])
        if cmd.get("reset"):
            self.camera.lookat[:] = self.model.stat.center
            self.camera.distance = max(2.5, 2.5*self.model.stat.extent)
            self.camera.azimuth, self.camera.elevation = 135, -25

    def pick(self, x, y):
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.options)
        point, geom, flex, skin = np.zeros(3), np.zeros(1, np.int32), np.zeros(1, np.int32), np.zeros(1, np.int32)
        body = mujoco.mjv_select(self.model, self.data, self.options, self.width/self.height,
                                number(x, 0, 1), 1-number(y, 0, 1), self.renderer.scene, point, geom, flex, skin)
        self.selected = max(0, body)

    def render(self):
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.options)
        if self.selected:
            for geom in self.renderer.scene.geoms[:self.renderer.scene.ngeom]:
                if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and 0 <= geom.objid < self.model.ngeom:
                    if self.model.geom_bodyid[geom.objid] == self.selected:
                        geom.rgba[:3] = .25*np.array(geom.rgba[:3]) + .75*np.array([.1, .8, .95])
        image = self.renderer.render()
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, "JPEG", quality=88, subsampling=0)
        return buffer.getvalue()

    def resize(self, width, height):
        width = int(number(width, 320, 1920))
        height = int(number(height, 240, 1200))
        if (width, height) != (self.width, self.height):
            renderer = mujoco.Renderer(self.model, height, width)
            self.renderer.close()
            self.renderer = renderer
            self.width, self.height = width, height

    def scene(self):
        m = self.model
        bodies = []
        for i in range(m.nbody):
            geoms = np.flatnonzero(m.geom_bodyid == i)
            depth, parent = 0, i
            while parent:
                depth += 1
                parent = int(m.body_parentid[parent])
            j = m.body_jntadr[i]
            free = j >= 0 and m.jnt_type[j] == 0
            position = self.data.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j]+3] if free else m.body_pos[i]
            bodies.append({"id": i, "name": m.body(i).name or f"body_{i}", "parent": int(m.body_parentid[i]),
                           "depth": depth, "mass": float(m.body_mass[i]), "position": position.tolist(),
                           "quaternion": (self.data.qpos[m.jnt_qposadr[j]+3:m.jnt_qposadr[j]+7] if free else m.body_quat[i]).tolist(),
                           "friction": float(m.geom_friction[geoms[0], 0]) if len(geoms) else 0,
                           "dynamic": bool(m.body_dofnum[i]), "geoms": len(geoms)})
        joints = []
        for j in range(m.njnt):
            if m.jnt_type[j] not in (2, 3): continue
            limited = bool(m.jnt_limited[j])
            limits = m.jnt_range[j].tolist() if limited else [-np.pi, np.pi] if m.jnt_type[j] == 3 else [-2, 2]
            joints.append({"id": j, "name": m.joint(j).name or f"joint_{j}", "body": int(m.jnt_bodyid[j]),
                           "lower": limits[0], "upper": limits[1], "unit": "rad" if m.jnt_type[j] == 3 else "m",
                           "qadr": int(m.jnt_qposadr[j]), "vadr": int(m.jnt_dofadr[j])})
        actuators = []
        for a in range(m.nu):
            limits = m.actuator_ctrlrange[a].tolist() if m.actuator_ctrllimited[a] else [-100, 100]
            kind = "position" if m.actuator_biastype[a] == 1 and m.actuator_biasprm[a, 1] != 0 else "velocity" if m.actuator_biastype[a] == 1 and m.actuator_biasprm[a, 2] != 0 else "motor"
            actuators.append({"id": a, "name": m.actuator(a).name or f"actuator_{a}", "lower": limits[0],
                              "upper": limits[1], "kind": kind, "pd": bool(self.pd_compatible(a))})
        return {"type": "scene", "revision": self.revision, "name": self.spec.modelname,
                "bodies": bodies, "joints": joints, "actuators": actuators,
                "sensors": [{"id": s, "name": m.sensor(s).name or f"sensor_{s}", "address": int(m.sensor_adr[s]),
                             "dimension": int(m.sensor_dim[s])} for s in range(m.nsensor)],
                "keyframes": [m.key(k).name or f"key_{k}" for k in range(m.nkey)],
                "gravity": m.opt.gravity.tolist(), "timestep": m.opt.timestep, "iterations": m.opt.iterations,
                "nq": m.nq, "nv": m.nv, "nu": m.nu}

    def status(self):
        contact_force = 0.0
        for i in range(self.data.ncon):
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            contact_force += abs(force[0])
        j = self.model.body_jntadr[self.selected]
        free = j >= 0 and self.model.jnt_type[j] == 0
        position = self.data.qpos[self.model.jnt_qposadr[j]:self.model.jnt_qposadr[j]+3] if free else self.model.body_pos[self.selected]
        return {"type": "status", "running": self.running, "time": float(self.data.time),
                "selected_position": position.tolist(),
                "selected_quaternion": (self.data.qpos[self.model.jnt_qposadr[j]+3:self.model.jnt_qposadr[j]+7] if free else self.model.body_quat[self.selected]).tolist(),
                "trajectory": self.trajectory["name"] if self.trajectory else "",
                "speed": self.speed, "rtf": self.actual_rtf, "selected": self.selected,
                "dirty": self.dirty, "project_path": self.project_path,
                "qpos": self.data.qpos.tolist(), "qvel": self.data.qvel.tolist(),
                "ctrl": self.data.ctrl.tolist(), "actuator_force": self.data.actuator_force.tolist(),
                "sensordata": self.data.sensordata.tolist(), "contacts": int(self.data.ncon),
                "contact_force": float(contact_force), "recording": self.recording,
                "record_count": len(self.record_times), "replaying": self.replay_states is not None,
                "replay_index": self.replay_index, "replay_count": len(self.replay_times) if self.replay_times is not None else 0,
                "can_undo": bool(self.undo_stack), "can_redo": bool(self.redo_stack),
                "controllers": self.controllers, "notice": self.last_notice,
                "warnings": int(sum(w.number for w in self.data.warning))}

    def close(self):
        if self.renderer:
            self.renderer.close()
