# 工作台架构与扩展契约

## 为什么保留 MuJoCo

此次重构改变主工作流和状态管理，不替换成熟的动力学求解器。工作台的编辑世界与批量训练后端分离，避免把多个 rollout 当成可编辑场景。场景由 MjSpec 描述，物理状态由 MjData 保存；编译成功且渲染器创建成功后才替换活动世界。

## 编辑事务

新增、删除、物体属性、物理配置、IMU 插入、多模型组合均先复制场景，再编译校验。失败时保留旧世界。成功时按稳定关节名和执行器名迁移物理状态，保留 12 次撤销历史。浮动关节位姿写入 qpos，不能仅修改 body.pos。关节姿态编辑也保存撤销快照。

## 时间与控制

物理步长属于场景，渲染目标 30 FPS，状态报告 10 Hz。目标实时倍率独立于三者。每次物理步前应用控制器；慢计算限制追赶窗口，报告实际 RTF，不跳过物理积分步伪造实时。

原生执行器输入保留 MuJoCo 模型语义。PD 只对关节传动、固定增益、无偏置、无执行器动态的标量电机开放，并按齿轮和增益转换，最后应用控制限幅。CSV 轨迹和手动控制互斥。仿真并不自动赋予导入机器人稳定站立和行走策略。

## 项目和录制

`.mjstudio`、`.mjrec` 均为 ZIP：`manifest.json`、`scene.xml`、`assets/`，录制另有 `states.npy` 和 `times.npy`。不会将归档路径直接解压到磁盘。读取限制解压总量、关闭 NumPy pickle，检查数据形状、有限值和时间顺序。保存通过同目录临时文件 + fsync + 原子替换。

源文件的 include 会在导入时解析；网格、纹理等资产改写为包内资源。模型组合通过 MjSpec 的复制附加模式，避免共享可变子场景。关键帧中的浮动基座平移单独处理。

## 本地控制 API

启动：`.venv/bin/python python/server.py --port 8765`，连接 `ws://127.0.0.1:8765`。UI 协议版本为 2。

请求为 JSON 对象，必须包含 `cmd`，可包含字符串 `request_id`。带请求 ID 的成功操作返回 `ack`，失败返回 `error`；场景变化返回 `scene`，状态返回 `status`，二进制帧为 JPEG。应检查应答后继续有依赖的操作。

```json
{"cmd":"load","builtin":"cartpole","request_id":"1"}
{"cmd":"control","id":0,"mode":"pd","target":0.2,"kp":20,"kd":5,"request_id":"2"}
{"cmd":"run","request_id":"3"}
{"cmd":"pause","request_id":"4"}
{"cmd":"save_project","path":"/absolute/path/experiment.mjstudio","request_id":"5"}
```

主要命令：

| 范围 | 命令 |
|---|---|
| 场景 | load, insert_model, add, remove, body_edit, add_imu, physics, undo, redo |
| 运行 | run, pause, step, reset, speed |
| 控制 | joint, control, force, load_trajectory, stop_trajectory |
| 显示 | select, pick, camera, view |
| 项目 | open_project, save_project |
| 记录 | record_start, record_stop, replay, seek, replay_exit, open_recording, export_recording, export_csv |

状态包含 qpos、qvel、ctrl、actuator_force、sensordata、接触统计、实际时间、RTF、录制状态和控制器配置。传感器地址、关节坐标地址与执行器名称由 scene 消息提供。外部控制程序可使用相同 API；该端口属于本地受信任控制接口，不能直接暴露到网络。

## 后续 Gazebo 类能力

需要独立设计与验证的模块：ROS 2 时间/TF/关节/传感器桥接，RGB-D/激光雷达，插件生命周期与进程隔离，模型资源库，SDF 兼容，多机任务与运动规划，训练管理，实时性能和大场景基准，Developer ID 签名及分发。

扩展时保持：物理线程单一所有者；UI 不持有 MuJoCo 指针；命令可追踪；结果记录使用仿真时间；任何模型变更需可回滚；传感器模型需声明单位、采样率与噪声模型。
