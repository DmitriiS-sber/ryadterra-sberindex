PYTHON ?= python
.PHONY: all verify smoke smoke-core package
all:
	$(PYTHON) src/prepare.py
	$(PYTHON) src/forecast.py
	$(PYTHON) src/default_baseline.py
	$(PYTHON) src/detect.py --config configs/validation.yaml
	$(PYTHON) src/news_warning.py
	$(PYTHON) src/future_warning.py
	$(PYTHON) src/forecast.py --availability-lag 1
	$(PYTHON) src/predict.py
	$(PYTHON) src/validation_diagnostics.py
	$(PYTHON) src/news_detector.py
	$(PYTHON) src/verify_validation.py
	$(PYTHON) src/verify.py
	$(PYTHON) src/build_materials.py
verify:
	$(PYTHON) src/verify.py
smoke:
	$(PYTHON) src/forecast.py --limit 8 --workers 2
smoke-core:
	$(PYTHON) src/forecast.py --limit 8 --workers 2 --skip-chronos
package:
	$(PYTHON) src/package.py
.PHONY: validation validation-numeric report verify-release
validation: validation-numeric report
validation-numeric:
	$(PYTHON) src/detect.py --config configs/validation.yaml
	$(PYTHON) src/future_warning.py
	$(PYTHON) src/validation_diagnostics.py
	$(PYTHON) src/news_detector.py
	$(PYTHON) src/verify.py
	$(PYTHON) src/verify_validation.py
	$(PYTHON) src/verify_release.py
report:
	$(PYTHON) src/build_materials.py
verify-release:
	$(PYTHON) src/verify_release.py

.PHONY: check-source smoke-lite
check-source:
	$(PYTHON) -m compileall -q src
	$(PYTHON) src/verify_release.py

smoke-lite:
	$(PYTHON) src/forecast.py --config configs/smoke.yaml --limit 8 --workers 1 --skip-chronos --output-dir results/smoke-lite
