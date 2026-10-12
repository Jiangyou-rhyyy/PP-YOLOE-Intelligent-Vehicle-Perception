# PP-YOLOE Intelligent Vehicle Perception Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a polished, reproducible GitHub repository for the existing PP-YOLOE+ intelligent-vehicle detector, including CPU inference, real model outputs, measured results, and an English README without retraining.

**Architecture:** Preserve the exported OpenVINO model as the runtime source of truth, move reusable image processing into a small `src/vehicle_perception` package, and expose image/directory and webcam workflows through thin CLI scripts. Keep large model binaries in a GitHub Release, generate Demo assets from supplied day/night scenes, and publish only after tests plus a clean-clone smoke test pass.

**Tech Stack:** Python 3.10-3.14, NumPy, OpenCV, OpenVINO Runtime, Pillow, PyYAML, pytest, Git, GitHub REST API.

**Spec:** `docs/superpowers/specs/2026-10-12-ppyoloe-vehicle-perception-design.md`

## Global Constraints

- Do not retrain or fine-tune the model.
- Present 40.23% mAP@0.5:0.95 and 17.59 FPS for PP-YOLOE+_s directly as measured project results.
- Preserve the full s/m/l comparison: `s = 40.23%, 17.59 FPS, 2.76 MB`; `m = 45.8%, 14.97 FPS, 28.7 MB`; `l = 47.9%, 10.88 FPS, 196 MB`.
- Use the public repository slug `PP-YOLOE-Intelligent-Vehicle-Perception` under `Jiangyou-rhyyy`.
- Keep BDD100K data out of Git and link to official dataset material.
- Put large model binaries in GitHub Release assets, not normal Git history.
- Generate all claimed detections with the supplied model; do not create illustrative boxes.
- Remove TensorFlow from inference dependencies.
- Publish English primary documentation and retain the Chinese project report as supporting material.
- Exclude the competition instruction DOCX from the public repository.

## Review Focus

- A missing or unreadable input path must produce a concise CLI error and non-zero exit, not a traceback from `cv2.imread`.
- A camera that cannot open or stops returning frames must exit cleanly and release resources.
- The model's 14 score channels versus 13 supplied labels must never cause an index error or silent wrong label.
- Empty detections must still produce an output image and a valid competition-format JSON file with an empty `objects` array.
- Running commands outside the repository root must still resolve explicit model, label, input, and output paths correctly.

---

### Task 1: Establish the publishable repository baseline

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `LICENSE`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `model/README.md`
- Create: `tests/test_repository_layout.py`
- Move: `code/doc/说明文档.pdf` to `docs/project-report-zh.pdf`
- Move: `code/labels.txt` to `model/labels.txt`
- Exclude: `附件4：智能车辆工程赛道说明.docx`
- Exclude after replacement: `code/detect.py`, `code/eval.py`, and the generic `code/train.py` copied from PaddleDetection
- Preserve as historical references under `docs/legacy/`: `code/doc/environment.yaml`, `code/doc/environment.txt`, and `code/doc/requirements.txt`
- Remove from publishable tree: large files under `model/` while preserving them in a release-staging directory outside Git

**Interfaces:**
- Consumes: Existing competition submission files and the approved design spec.
- Produces: Installable package metadata, dependency declarations, legal notices, tracked small model metadata, and a repository tree safe to commit.

- [ ] **Step 1: Write repository-layout tests**

Create tests named `test_no_tracked_file_exceeds_github_limit`, `test_competition_docx_is_excluded`, `test_required_project_files_exist`, and `test_model_readme_lists_required_release_files`. Assert a 20 MiB project policy limit for normal Git files and require `model.xml` plus `model.bin` in the release manifest.

- [ ] **Step 2: Run the layout tests and verify they fail**

Run: `pytest tests/test_repository_layout.py -v`

Expected: FAIL because the baseline files and release manifest do not exist and large binaries remain in the publishable tree.

- [ ] **Step 3: Add the minimum repository baseline**

Use a `src`-layout `pyproject.toml`; require Python `>=3.10`; declare runtime dependencies for NumPy, OpenCV, OpenVINO, Pillow, and PyYAML; declare pytest in `requirements-dev.txt`. Confirm the final OpenVINO version by a real model smoke test before freezing it. Add Apache-2.0 licensing and explicit PaddleDetection/PP-YOLOE/BDD100K/OpenVINO notices. Preserve the old environment exports only under `docs/legacy/`, clearly separated from the supported inference installation.

- [ ] **Step 4: Stage large models outside Git and normalize documentation paths**

Create `work/release-assets/openvino/` and `work/release-assets/paddle/` beside the repository, copy the corresponding model files there, then remove those binaries from the publishable tree. Keep `infer_cfg.yml`, the normalized label file, and `model/README.md` tracked.

- [ ] **Step 5: Run the layout tests**

Run: `pytest tests/test_repository_layout.py -v`

Expected: PASS.

- [ ] **Step 6: Commit the baseline**

Run: `git add .gitignore pyproject.toml requirements*.txt LICENSE THIRD_PARTY_NOTICES.md model/README.md model/infer_cfg.yml model/labels.txt docs/project-report-zh.pdf tests/test_repository_layout.py && git commit -m "chore: establish publishable project baseline"`

### Task 2: Implement preprocessing and postprocessing with tests

**Files:**
- Create: `src/vehicle_perception/__init__.py`
- Create: `src/vehicle_perception/types.py`
- Create: `src/vehicle_perception/preprocess.py`
- Create: `src/vehicle_perception/postprocess.py`
- Create: `src/vehicle_perception/visualization.py`
- Create: `tests/test_preprocess.py`
- Create: `tests/test_postprocess.py`
- Create: `tests/test_visualization.py`

**Interfaces:**
- Consumes: BGR `numpy.ndarray` images, raw `[1, 8400, 4]` boxes, raw `[1, C, 8400]` scores, and label strings.
- Produces: `ImageTransform`, `Detection`, `preprocess_image(image, input_size=640)`, `postprocess_detections(boxes, scores, labels, confidence_threshold=0.5, iou_threshold=0.35, extra_channel_policy="error")`, and `draw_detections(image, detections)`.

- [ ] **Step 1: Write failing preprocessing tests**

Assert that `preprocess_image` returns `float32` NCHW `[1, 3, 640, 640]`, RGB-normalized values in `[0, 1]`, and metadata that maps coordinates back to the original image for both landscape and portrait inputs. Add a test that an empty image raises `ValueError`.

- [ ] **Step 2: Run preprocessing tests and verify failure**

Run: `pytest tests/test_preprocess.py -v`

Expected: FAIL because the package does not exist.

- [ ] **Step 3: Implement preprocessing types and function**

Define immutable dataclasses `ImageTransform(original_height: int, original_width: int, padded_size: int, input_size: int)` and `Detection(class_id: int, label: str, score: float, box: tuple[float, float, float, float])`. Preserve the existing top-left square-padding and nearest-neighbor resize behavior.

- [ ] **Step 4: Run preprocessing tests**

Run: `pytest tests/test_preprocess.py -v`

Expected: PASS.

- [ ] **Step 5: Write failing postprocessing and visualization tests**

Cover confidence filtering, class-aware NMS, coordinate clipping, empty detections, 13-channel scores, 14-channel scores under `error` and explicit extra-channel policies, unknown class IDs, and deterministic label/color drawing without modifying the input array.

- [ ] **Step 6: Implement postprocessing and drawing**

Use OpenCV `dnn.NMSBoxes` per class or an equivalent NumPy NMS. Never infer the extra-channel meaning silently: `error` rejects a mismatch, while a verified policy supplied by the model runner handles the 14th channel explicitly.

- [ ] **Step 7: Run the processing test suite**

Run: `pytest tests/test_preprocess.py tests/test_postprocess.py tests/test_visualization.py -v`

Expected: PASS.

- [ ] **Step 8: Commit the processing core**

Run: `git add src/vehicle_perception tests/test_preprocess.py tests/test_postprocess.py tests/test_visualization.py && git commit -m "feat: add tested detection processing core"`

### Task 3: Add the OpenVINO runner and resolve model output mapping

**Files:**
- Create: `src/vehicle_perception/model.py`
- Create: `src/vehicle_perception/io.py`
- Create: `tests/test_model.py`
- Create: `tests/test_io.py`
- Modify: `model/README.md`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: OpenVINO `model.xml` and sibling `model.bin`, processed input arrays, labels, and the supplied day/night images.
- Produces: `OpenVINODetector(model_path: Path, labels: Sequence[str], device: str = "CPU")`, `predict(image) -> list[Detection]`, `load_labels(path) -> list[str]`, `write_competition_json(path, image_name, detections, timestamp)`, `smoke_test(model_path: Path, image_path: Path, labels_path: Path) -> dict[str, object]`, and verified score-channel policy metadata.

- [ ] **Step 1: Create an isolated Python environment and install candidate runtime dependencies**

Use Python 3.10+ and install the current stable OpenVINO release plus project dependencies. Request network permission before installation. Do not install TensorFlow.

- [ ] **Step 2: Write failing model and I/O tests**

Mock OpenVINO for input/output ordering and device configuration. Test label loading with whitespace-containing labels, JSON output with detections and no detections, explicit paths from a different working directory, missing model files, and mismatch errors for unverified score channels.

- [ ] **Step 3: Run the unit tests and verify failure**

Run: `pytest tests/test_model.py tests/test_io.py -v`

Expected: FAIL because the runner and I/O helpers do not exist.

- [ ] **Step 4: Implement the runner and I/O helpers**

Read outputs by shape rather than positional assumptions: boxes contain a trailing dimension of 4 and scores contain the 8,400-candidate dimension. Keep the score-channel policy explicit in `OpenVINODetector`.

- [ ] **Step 5: Run a real supplied-model diagnostic on both road scenes**

Record input names, output names/shapes, per-channel maximums, predicted class IDs above threshold, and whether channel 13 ever produces a candidate. Resolve the 14-to-13 mapping only from this evidence and model/report metadata; otherwise label it `unknown` and exclude it from public class claims.

- [ ] **Step 6: Freeze the verified dependency versions and mapping behavior**

Update dependency files and `model/README.md` with the runtime version and explicit channel behavior proven by the smoke test.

- [ ] **Step 7: Run unit and real-model tests**

Run: `pytest tests/test_model.py tests/test_io.py -v`

Run: `python -m vehicle_perception.model --smoke-test --model ../release-assets/openvino/model.xml --labels model/labels.txt --image assets/source/day.jpg`

Expected: Unit tests PASS; smoke test loads on CPU and returns structured detections without label errors.

- [ ] **Step 8: Commit the model integration**

Run: `git add src/vehicle_perception/model.py src/vehicle_perception/io.py tests/test_model.py tests/test_io.py model/README.md pyproject.toml requirements.txt && git commit -m "feat: integrate verified OpenVINO inference"`

### Task 4: Build image and webcam command-line workflows

**Files:**
- Create: `scripts/detect_images.py`
- Create: `scripts/detect_webcam.py`
- Create: `tests/test_detect_images_cli.py`
- Create: `tests/test_detect_webcam_cli.py`

**Interfaces:**
- Consumes: Task 3's `OpenVINODetector`, path and JSON helpers, explicit CLI arguments.
- Produces: image/directory inference with annotated image and optional JSON output; asynchronous webcam display with clean shutdown.

- [ ] **Step 1: Write failing CLI tests**

Test `--help`, missing input, unreadable image, automatic output-directory creation, single-image and directory modes, empty detections, JSON output, invocation outside repo root, camera-open failure, dropped frame, and ESC shutdown with mocked OpenCV/OpenVINO objects.

- [ ] **Step 2: Run CLI tests and verify failure**

Run: `pytest tests/test_detect_images_cli.py tests/test_detect_webcam_cli.py -v`

Expected: FAIL because the scripts do not exist.

- [ ] **Step 3: Implement `detect_images.py`**

Arguments: `--model`, `--labels`, mutually exclusive `--image` or `--image-dir`, `--output-dir`, `--json-dir`, `--device`, `--confidence`, and `--iou`. Return non-zero with concise messages for user input errors.

- [ ] **Step 4: Implement `detect_webcam.py`**

Arguments: `--model`, `--labels`, `--camera`, `--device`, `--confidence`, `--iou`, and `--window-size`. Preserve double-buffered asynchronous inference and guarantee `release()` plus `destroyAllWindows()` in `finally`.

- [ ] **Step 5: Run CLI and regression tests**

Run: `pytest -v`

Expected: PASS.

- [ ] **Step 6: Commit the workflows**

Run: `git add scripts tests/test_detect_images_cli.py tests/test_detect_webcam_cli.py && git commit -m "feat: add image and webcam inference commands"`

### Task 5: Generate and verify real Demo and benchmark assets

**Files:**
- Create: `assets/source/day.jpg`
- Create: `assets/source/night.jpg`
- Create: `assets/demo-day.jpg`
- Create: `assets/demo-night.jpg`
- Create: `assets/demo.gif`
- Create: `assets/benchmark.svg`
- Create: `scripts/build_assets.py`
- Create: `tests/test_assets.py`

**Interfaces:**
- Consumes: The supplied day/night road scenes, Task 4 image CLI, and the fixed s/m/l benchmark values.
- Produces: actual annotated screenshots, a compact looping GIF, and an SVG comparison graphic used by README.

- [ ] **Step 1: Write failing asset tests**

Assert both source images are readable; generated screenshots match source dimensions; the GIF has at least two frames and is below 8 MiB; the SVG contains all three model names and exact benchmark values; generated outputs contain non-identical pixels versus inputs.

- [ ] **Step 2: Run asset tests and verify failure**

Run: `pytest tests/test_assets.py -v`

Expected: FAIL because assets and builder do not exist.

- [ ] **Step 3: Implement deterministic asset generation**

Extract the two supplied scenes, run the existing model, build a two-frame GIF with readable dwell time, and create a repository-native SVG comparing mAP, FPS, and size. Do not draw or alter detections after inference except for label visualization.

- [ ] **Step 4: Run asset tests and inspect every visual**

Run: `pytest tests/test_assets.py -v`

Expected: PASS. Open both JPGs, the GIF frames, and SVG rendering; verify boxes, labels, contrast, cropping, and legibility.

- [ ] **Step 5: Commit Demo assets**

Run: `git add assets scripts/build_assets.py tests/test_assets.py && git commit -m "docs: add real detection demos and benchmarks"`

### Task 6: Write and validate the English repository documentation

**Files:**
- Create: `README.md`
- Create: `data/images/.gitkeep`
- Create: `data/results/.gitkeep`
- Create: `scripts/check_readme.py`
- Create: `tests/test_readme.py`
- Modify: `THIRD_PARTY_NOTICES.md`

**Interfaces:**
- Consumes: Verified CLI commands, generated assets, exact benchmark values, official source URLs, and target project structure.
- Produces: Final English README with Overview, Core Features, Demo, Quick Start, Project Structure, Results, and Dataset & Acknowledgements sections.

- [ ] **Step 1: Write failing README tests**

Require the exact title; required section headings; all benchmark values; clone/install/model-download/image/webcam commands; links to real local assets; BDD100K, PaddleDetection, PP-YOLOE paper, and OpenVINO official links; no local absolute paths, placeholder text, obsolete contact information, or unsupported accuracy claims.

- [ ] **Step 2: Run README tests and verify failure**

Run: `pytest tests/test_readme.py -v`

Expected: FAIL because README does not exist.

- [ ] **Step 3: Draft the README and supporting notices**

Lead with the real GIF, describe the 13 named traffic classes and verified handling of any extra score channel, present Results directly as project measurements, and separate inference Quick Start from optional historical training guidance.

- [ ] **Step 4: Validate links, commands, prose, and rendered Markdown structure**

Run: `python scripts/check_readme.py && pytest tests/test_readme.py -v`

Expected: PASS with zero missing paths or prohibited content.

- [ ] **Step 5: Commit documentation**

Run: `git add README.md data scripts/check_readme.py tests/test_readme.py THIRD_PARTY_NOTICES.md && git commit -m "docs: publish complete English project guide"`

### Task 7: Package models, verify the full repository, and publish

**Files:**
- Create: `scripts/download_models.py`
- Create: `tests/test_download_models.py`
- Create outside Git: `work/release-assets/ppyoloe-openvino-model.zip`
- Optionally create outside Git: `work/release-assets/ppyoloe-paddle-model.zip`
- Modify: `README.md`
- Modify: `model/README.md`

**Interfaces:**
- Consumes: Verified repository, GitHub authentication from Git Credential Manager, and staged model files.
- Produces: Public GitHub repository, release assets, working download helper, synchronized original local project, and clean-clone verification evidence.

- [ ] **Step 1: Write failing model-download tests**

Mock HTTP and test latest-release URL construction, checksum verification, safe ZIP extraction, an existing valid model, corrupt downloads, and network failure without partial model files.

- [ ] **Step 2: Run download tests and verify failure**

Run: `pytest tests/test_download_models.py -v`

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement model packaging and download helper**

Package the OpenVINO XML/BIN pair with SHA-256 checksums. Make the helper download the named release asset from `Jiangyou-rhyyy/PP-YOLOE-Intelligent-Vehicle-Perception`, validate checksum, and extract only expected filenames.

- [ ] **Step 4: Run complete local verification**

Run: `pytest -v`

Run: `python scripts/check_readme.py`

Run: `git diff --check && git status --short`

Run secret and large-file scans; verify the real model on both Demo images.

Expected: All tests PASS; no secrets; no tracked file exceeds policy; only intended changes remain.

- [ ] **Step 5: Commit the release workflow**

Run: `git add scripts/download_models.py tests/test_download_models.py README.md model/README.md && git commit -m "feat: add verified model release workflow"`

- [ ] **Step 6: Perform final independent code review**

Use `superpowers:requesting-code-review`; resolve every blocking issue and rerun the full verification suite.

- [ ] **Step 7: Create and push the public GitHub repository**

Request network permission, create the repository through GitHub's official API using the existing credential-manager token without printing it, set `origin`, and push `main`.

- [ ] **Step 8: Publish model release assets**

Create release tag `v1.0.0-models`, upload the OpenVINO bundle and checksum, then confirm public download URLs.

- [ ] **Step 9: Run a clean-clone end-to-end smoke test**

Clone into a fresh temporary directory, install dependencies, run the download helper, infer on one Demo source image, and confirm annotated image plus JSON output.

- [ ] **Step 10: Synchronize the verified tree to the original project directory**

Request write permission for `D:\“人工智能+”交叉学科创新应用大赛`, verify the resolved source and destination, preserve a backup of files that will be replaced, and copy only the final tracked project files plus local model assets. Do not copy `.git` unless explicitly required.

- [ ] **Step 11: Report final repository URL and evidence**

Include the GitHub URL, release URL, local project path, final commit, test counts, Demo status, and any environment-specific limitation found during clean-clone verification.
