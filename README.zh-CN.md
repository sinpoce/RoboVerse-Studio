# RoboVerse Studio

基于 SwiftUI 和 MuJoCo 的 macOS 机器人仿真与强化学习工作台。

[下载](https://github.com/sinpoce/RoboVerse-Studio/releases/latest) · [English](README.md) · [训练指南](docs/CUSTOM_TRAINING.md)

![场景工作台](docs/screenshots/scene-workbench.png)

## 功能

- 导入 MJCF/XML 和受支持的 URDF 模型，编辑场景、物体属性、关节和执行器。
- 运行、暂停和单步仿真，查看接触、IMU 传感器和实时曲线。
- 保存便携项目，录制与回放实验，导出 CSV 数据。
- 配置训练任务，查看训练曲线，保存与恢复检查点，评估策略。
- 支持英文和简体中文，默认使用英文。

内置 Unitree G1、倒立摆和双摆模型。训练工作区支持任务与算法插件，以及用于 Go1/H1 任务的 [Stage-Wise-CMORL](https://github.com/rllab-snu/Stage-Wise-CMORL) 适配器。导入项目及其资源需单独下载。

![训练工作区](docs/screenshots/training-workspace.png)

![机器人检查器](docs/screenshots/robot-inspector.png)

## 安装

**Apple Silicon · macOS 14 或更高版本**

从 [Releases](https://github.com/sinpoce/RoboVerse-Studio/releases/latest) 下载 ZIP，解压后将 **RoboVerse Studio.app** 移至“应用程序”。安装包自带 Python、MuJoCo 和 PyTorch。

应用使用 ad hoc 签名，尚未经过 Apple 公证。如首次启动被系统阻止，可前往“系统设置 → 隐私与安全性 → 仍要打开”。下载附件提供 SHA-256 校验文件。

## 使用

从模型库选择机器人或导入自己的模型。拖动旋转视角，Shift＋拖动平移，滚动缩放。在“关节”和“控制”中配置机器人，用“录制”和“回放”保存实验过程。场景可保存为 `.mjstudio` 项目。

切换到“训练”后可导入兼容的任务配置或项目。支持的适配器与参数说明见[训练指南](docs/CUSTOM_TRAINING.md)。软件不包含预训练策略。

## 构建

需要 macOS、Swift 5.9+ 和 Python 3.11。

```bash
git clone https://github.com/sinpoce/RoboVerse-Studio.git
cd RoboVerse-Studio
PYTHON=python3.11 make setup
make run
```

`make app` 将独立应用构建至 `dist/`。打包需使用可迁移的 CPython，例如 uv 管理的 Python。

## 许可

[Apache-2.0](LICENSE)。模型与依赖遵循各自上游许可，详见[第三方声明](THIRD_PARTY_NOTICES.md)。

使用 [MuJoCo](https://github.com/google-deepmind/mujoco)、[mjbatch](https://github.com/kevinzakka/mjbatch) 和 [PyTorch](https://github.com/pytorch/pytorch)。G1 模型来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1)。
