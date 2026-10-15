# Third-party notices

RoboVerse Studio includes or uses the following upstream projects. Their original license files and package metadata remain applicable and are preserved in the source model folder or bundled Python runtime.

| Component | License / notice | Source |
| --- | --- | --- |
| MuJoCo | Apache-2.0 | https://github.com/google-deepmind/mujoco |
| mjbatch | Apache-2.0 | https://github.com/kevinzakka/mjbatch |
| Unitree G1 model | BSD-3-Clause; see `models/unitree_g1/LICENSE` | https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1 |
| CPython | Python Software Foundation license and bundled notices | https://www.python.org/ |
| PyTorch | BSD-style license and third-party notices | https://github.com/pytorch/pytorch |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy |
| Pillow | HPND | https://github.com/python-pillow/Pillow |
| websockets | BSD-3-Clause | https://github.com/python-websockets/websockets |
| PyYAML | MIT | https://github.com/yaml/pyyaml |

Stage-Wise-CMORL source, robot assets, checkpoints, and training runs are **not distributed in this repository or app release**. Download that project separately and follow its upstream license, asset notices, and research citation requirements. The tested integration used commit `062ddc1f2feabf94504e4d5ebc6835cb11c8bf92`.

The release runtime retains dependency `.dist-info` license directories, including notices for transitive dependencies. Distribution of a model does not imply endorsement by its manufacturer.
