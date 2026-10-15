# Development

Run `make setup` once, then `make run` to open the scene workbench. The Training workspace launches its own task training service when selected.

- `StudioModel.swift` and `WorkbenchView.swift`: scene editing, simulation, recording, replay, and inspector.
- `TrainingModel.swift`, `TrainingWorkspaceView.swift`, and `TrainingClient.swift`: training configuration, telemetry, checkpoints, and evaluation.
- `python/server.py`: scene service.
- `examples/generic_train_server.py` and `python/training_transport.py`: task training service and loopback WebSocket transport.

`make test` runs the physics, persistence, recording, task, algorithm, and service checks. Set `ROBOVERSE_STAGEWISE_PROJECT` to a Stage-Wise-CMORL checkout to include adapter and training-service checks.

```bash
swiftc Sources/MjbatchStudio/Localization.swift scripts/check_localization.swift -o /tmp/roboverse-localization-check
/tmp/roboverse-localization-check Sources/MjbatchStudio/Resources/translations.json
make app
ROBOVERSE_STAGEWISE_PROJECT=/absolute/path/Stage-Wise-CMORL .venv/bin/python scripts/verify_app.py
```

Quit the app before rebuilding its bundle. `make app` uses a relocatable Python runtime and leaves the preceding bundle in `dist/` for local recovery. The verification script copies the bundle to a temporary directory and runs it without source paths.
