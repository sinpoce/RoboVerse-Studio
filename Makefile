.PHONY: setup run train-custom train-custom-ui xcode clean app test

setup:
	bash scripts/setup_python.sh

run:
	swift run RoboVerseStudio

train-custom:
	$(CURDIR)/.venv/bin/python examples/train_custom.py \
		$(if $(TASK),--task "$(TASK)",) \
		$(if $(MODEL),--model "$(MODEL)",) \
		--config "$(or $(CONFIG),examples/configs/g1_backflip.json)" \
		$(if $(ALGORITHM),--algorithm "$(ALGORITHM)",) \
		--checkpoint "$(or $(CHECKPOINT),runs/policy.pt)"

train-custom-ui:
	MJBATCH_STUDIO_PYTHON="$(CURDIR)/.venv/bin/python" \
	MJBATCH_STUDIO_SERVER="$(CURDIR)/examples/generic_train_server.py" \
	$(if $(TASK),MJBATCH_STUDIO_TASK="$(TASK)",) \
	$(if $(MODEL),MJBATCH_STUDIO_MODEL="$(MODEL)",) \
	MJBATCH_STUDIO_TASK_CONFIG="$(or $(CONFIG),$(CURDIR)/examples/configs/g1_backflip.json)" \
	MJBATCH_STUDIO_ALGORITHM="$(or $(ALGORITHM),)" \
	MJBATCH_STUDIO_CHECKPOINT="$(or $(CHECKPOINT),$(CURDIR)/runs/policy.pt)" \
	swift run RoboVerseStudio

xcode:
	xcodegen generate

app:
	.venv/bin/python scripts/package_app.py

test:
	.venv/bin/python -m unittest discover -s tests -v

clean:
	rm -rf .build RoboVerseStudio.xcodeproj
