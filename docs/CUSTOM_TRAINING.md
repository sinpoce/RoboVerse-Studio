# Custom models, tasks and algorithms

The editable workbench and the learning backend use the same MuJoCo XML but
have different responsibilities. The workbench validates geometry, contacts,
controllers and sensors. A task plugin defines what should be learned. An
algorithm plugin consumes the task's observation/action/reward stream, so PPO
is only one selectable implementation.

## What a user can customise

- Any MuJoCo MJCF entry file, including an XML that includes meshes and
  textures in a sibling `assets/` directory.
- The action mode and scale. The G1 example uses 29 position targets; a custom
  model can expose velocity, torque or raw actuator controls.
- The observation vector returned by `observe_task()`.
- Reward terms, success and termination conditions.
- Reset noise, episode duration, control decimation and domain randomisation.
- Algorithm-specific settings. The built-in PPO accepts horizon, discount,
  GAE, clipping, learning rate, entropy, network widths, number of
  environments and checkpoint interval.

## Task package layout

```text
my_task_project/
├── model/robot.xml
├── model/assets/...
├── tasks/my_task.py
├── algorithms/my_algorithm.py
├── configs/my_task.json
└── runs/
```

`tasks/my_task.py` exports `create_task(model_path, num_envs, config)`. Copy
`examples/tasks/template.py` and implement the three hooks:

```python
observe_task() -> (num_envs, obs_dim)
reward_task(action) -> (reward_vector, term_dictionary)
terminated_task() -> (terminated_vector, success_vector)
```

The base class applies clipped actions, steps all environments, handles time
outs and resets completed episodes. A plugin can override `reset_task()` to
randomise initial state and physical parameters.

## Algorithm plug-ins

Every custom algorithm exports one function with this signature:

```python
ALGORITHM_ID = "train_my_algorithm"
LABEL = "My algorithm"

def train(task, config, checkpoint, on_iteration=None, stop=None):
    # task.reset(), task.step(action) and task.metadata() are the stable API.
    # Call on_iteration({...}) with reward/success/terms for live charts.
    # Return a dict containing at least iteration/reward when possible.
    ...
```

Add its import path to the config:

```json
{
  "algorithm": "my_algorithms:train",
  "algorithms": ["ppo", "my_algorithms:train"],
  "algorithm_config": {
    "train_my_algorithm": {"iterations": 200, "seed": 7}
  }
}
```

The app discovers the entry point and exposes it in the Algorithm picker.
`examples/algorithms/random_search.py` is a working non-PPO plug-in used to
validate the interface. It is a baseline for integration testing, not a
backflip solver. A production algorithm can use PyTorch, JAX, imitation data,
model predictive control or any other dependency installed in the training
environment.

## JSON configuration

`examples/configs/g1_backflip.json` is a complete example. The `task` section
contains task-specific values. `algorithm` selects the initial algorithm and
`algorithms` lists every algorithm that should appear in the RoboVerse Studio
menu. The `ppo` section contains PPO values. Other algorithms read their own
`algorithm_config` object.

Run the sample from the repository root:

```bash
make setup
make train-custom ALGORITHM=ppo
```

Or select a custom task, model and algorithm:

```bash
make train-custom \
  TASK=my_task:create_task \
  MODEL=/absolute/path/to/robot.xml \
  CONFIG=/absolute/path/to/my_task.json \
  ALGORITHM=my_algorithms:train \
  CHECKPOINT=/absolute/path/to/runs/my_task.pt
```

The standalone app includes training in the main workbench: switch to **训练**
in the toolbar (⌘⇧T). Load the JSON task configuration, choose an algorithm,
edit parameters and checkpoint path in the right inspector, then click
**开始训练**. Full JSON parameters are available in **完整算法参数**. The
panel plots the metrics reported by the actual algorithm. **停止并保存**
requests a stop; wait for the saved checkpoint message before closing.

For development, `make train-custom-ui` accepts the same `TASK`, `MODEL`,
`CONFIG`, `ALGORITHM` and `CHECKPOINT` overrides.

The `TASK` and algorithm modules must be importable from the current project
or `examples/`. Each algorithm owns its checkpoint format; PPO writes a
PyTorch policy checkpoint and the sample random-search plug-in writes a pickle
checkpoint.

## Importing an existing project

### Stage-Wise-CMORL on macOS

The bundled app runs the downloaded `Stage-Wise-CMORL-main` project locally:

1. Switch to **训练 → 导入已有项目…** and select the project folder.
2. In the left task list, select **Comoppo · Go1 Backflip · MuJoCo**.
3. In the right inspector, for a first smoke run set 并行环境 (`num_envs`) to
   `2`, 每次采样步数 (`horizon`) to `128`, 训练迭代次数 (`iterations`) to `3`, and
   恢复检查点步数 (`model_num`) to `0`. Set a checkpoint path in a writable directory.
4. Click **开始训练**. Parameters, environment setup and launch are applied together.
5. Observe live MuJoCo frames, original reward/cost terms and learning losses.
   A completion message identifies the saved checkpoint. Increase `iterations`
   for a real experiment; three iterations do not teach a backflip.

Supported tasks: Go1 backflip, sideflip, sideroll and twohand; H1 backflip
and twohand. The adapter executes their original observation, action, PD
control, reward, cost, stage-transition and termination code. It runs the
project's original COMOPPO actor, reward/cost critics, optimizer and
normalizers. MuJoCo supplies dynamics, joint states and contact forces
through an explicit adapter. Isaac Gym and CUDA are not required.

This currently runs on CPU. MuJoCo contact/solver behavior differs from
PhysX, so original paper results and pretrained PhysX policy behavior are
not guaranteed. PhysX material domain randomization is not ported; task YAML
files with `randomize.is_randomized: true` are rejected explicitly. The
original source files and assets are read, not rewritten.

Use `algorithm_overrides` to override original YAML hyperparameters, for
example `{"actor_lr": 0.00003, "n_actor_iters": 20}`. Overrides affect the
project's own algorithm, rather than selecting the built-in PPO.

Checkpoints have two parts: `policy.pt` and its adjacent `policy.run/`
directory. Keep both. The directory includes numbered checkpoints, optimizer
state, normalizers, and `experiment.json` with source hashes and the actual
completed step count. Periodic saves use `checkpoint_every` (iterations).
After **停止并保存**, the app sets `model_num` to the saved step count;
**开始训练** continues from that policy. After restarting the app, import the
folder again, set the same checkpoint path, and set `model_num` to
`total_steps` from `experiment.json`. A new run uses `model_num: 0` and a
new checkpoint path. Resume restores parameters, optimizer and normalizers;
it starts fresh physics episodes and replay buffers, not a bit-exact replay
of an interrupted run.

For student imitation, select **Student · Go1 Backflip · MuJoCo**, set
`teacher_run` to a COMOPPO `policy.run/` directory for the same task, and use
a separate student checkpoint path. The original student algorithm learns
from that teacher's actions. An untrained teacher will not provide a useful
backflip demonstration.

The CLI accepts the project folder directly:

```bash
.venv/bin/python examples/train_custom.py \
  --config "$HOME/Downloads/Stage-Wise-CMORL-main" \
  --algorithm train_project_comoppo_go1_backflip \
  --num-envs 2 --iterations 3 \
  --checkpoint runs/go1_backflip/policy.pt
```

### Other projects

The importer scans model assets, task YAML files and algorithm source files.
Only projects with an implemented physics/training adapter can run directly;
arbitrary Isaac Gym, Isaac Sim, ROS or Gazebo programs are not automatically
translated to MuJoCo. Other projects can expose a task and algorithm through
the plugin contract above. Additional dependencies must be installed in the
chosen Python environment; the app does not install a downloaded project's
requirements automatically.

To save a reusable scan manifest explicitly:

```bash
python examples/import_project.py \
  --project /Users/<user>/Downloads/Stage-Wise-CMORL-main \
  --output /Users/<user>/Downloads/Stage-Wise-CMORL-main/roboverse.project.json
```

The manifest is also accepted as a training configuration. Folder import
does not write a manifest into the source project.

## Back in the workbench

Before training, open the same XML in the workbench and validate the stand/reset
pose, actuator ranges, floor contacts, sensors and a short reference trajectory.
The CSV trajectory feature is a debugging and imitation aid; it is not a
replacement for the task plugin. Stage-Wise backflip checkpoints can be loaded and evaluated in the training
workspace as described below. Hardware deployment is not implemented.

## Independent policy evaluation

For imported Stage-Wise Go1/H1 backflip tasks, **加载检查点…** validates the
robot/task, algorithm, numbered policy and companion normalizers, and restores
the saved training step and hyperparameters. Preserve both `policy.pt` and
`policy.run/`; a new run must use a new output path.

**评估策略** runs five deterministic episodes with independent seeds, streams
the policy-controlled MuJoCo frames, and reports a measured success rate.
Success requires backward rotation of at least 330 degrees, inversion, real
flight, and an upright foot-supported landing sustained for 0.25 seconds,
without a base/hip collision or task failure. Merely standing, reaching a task
stage or obtaining a high training reward does not count.

**查看评估报告与回放** opens the evaluation folder containing `evaluation.json`,
per-episode state traces and CSV files, a GIF, and a portable `episode-1.mjrec`
recording with embedded robot assets. **在场景中回放** opens that recording
directly in the scene workbench; `robot.mjstudio` contains its starting state.
Training metrics are persisted in `policy.run/metrics.jsonl`. Evaluating a
checkpoint never updates the network or optimizer. Other task/algorithm
formats still require their own evaluation adapter.
