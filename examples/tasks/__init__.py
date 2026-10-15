# SPDX-License-Identifier: Apache-2.0
"""User task plugins for the configurable algorithm runner.

Every module exposes ``create_task(model_path, num_envs, config)``.  A user can
drop another module into this directory (or point the runner at any importable
module) without changing the simulator or a training algorithm.
"""
