# PP-YOLOE Intelligent Vehicle Perception Repository Design

## Purpose

Turn the existing competition submission into a polished, reproducible English-language GitHub repository without retraining the detector. The repository will present the trained PP-YOLOE+ system, preserve the measured project results, provide CPU inference through OpenVINO, and include real visual outputs generated from the supplied model and road-scene images.

## Repository identity

- GitHub owner: `Jiangyou-rhyyy`
- Repository slug: `PP-YOLOE-Intelligent-Vehicle-Perception`
- README title: `Intelligent Vehicle Environment Perception and Real-Time Object Detection System Based on PP-YOLOE+`
- Visibility: public, matching the previously published project
- Primary language: English

The long descriptive title will be used for presentation, while the shorter slug keeps clone URLs and commands readable.

## Goals

1. Make image-folder inference and webcam inference straightforward on a CPU machine.
2. Present the existing project benchmark as measured results:
   - PP-YOLOE+_s: 40.23% mAP@0.5:0.95, 17.59 FPS, 2.76 MB model size
   - PP-YOLOE+_m: 45.8% mAP@0.5:0.95, 14.97 FPS, 28.7 MB model size
   - PP-YOLOE+_l: 47.9% mAP@0.5:0.95, 10.88 FPS, 196 MB model size
3. Explain why PP-YOLOE+_s was selected as the speed-accuracy-size trade-off.
4. Generate actual day and night detection screenshots and a compact GIF from the supplied model.
5. Document dataset preparation and project attribution using official PP-YOLOE, PaddleDetection, OpenVINO, and BDD100K sources.
6. Publish a clean repository under the user's existing GitHub account.

## Non-goals

- Retraining or fine-tuning the detector.
- Reproducing the full 70,000-image training run.
- Claiming new accuracy measurements beyond the existing project benchmark.
- Uploading the BDD100K dataset.
- Expanding the project into planning, control, tracking, or full autonomous-driving functionality.

## Selected implementation approach

Use a documentation-first reproducible showcase with targeted code hardening.

The existing exported OpenVINO model remains the inference source of truth. The inference code will be reorganized into reusable preprocessing, postprocessing, drawing, and model-runner functions. Thin command-line entry points will support image or directory inference and webcam inference. This keeps the project recognizable while removing fragile working-directory assumptions and unnecessary runtime dependencies.

Alternative approaches were rejected:

- Documentation-only packaging would leave hard-coded paths, missing output-directory creation, and a weak demo workflow.
- A full framework migration or retraining effort would add cost and compatibility risk without improving the requested repository presentation.

## Source layout

The target repository will use the following structure:

```text
.
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- assets/
|   |-- demo.gif
|   |-- demo-day.jpg
|   |-- demo-night.jpg
|   `-- benchmark.svg
|-- data/
|   |-- images/
|   `-- results/
|-- docs/
|   |-- project-report-zh.pdf
|   `-- superpowers/specs/
|-- model/
|   |-- infer_cfg.yml
|   |-- labels.txt
|   `-- README.md
|-- src/
|   `-- vehicle_perception/
|       |-- __init__.py
|       |-- inference.py
|       |-- postprocess.py
|       `-- visualization.py
|-- scripts/
|   |-- detect_images.py
|   |-- detect_webcam.py
|   `-- download_models.py
`-- tests/
    |-- test_preprocess.py
    |-- test_postprocess.py
    `-- test_paths.py
```

The original competition report will be retained as supporting documentation. The competition instruction DOCX will not be published because it contains obsolete contact details, a group QR code, and event-specific distribution information that do not help repository users.

## Model distribution

Large binary model files will be packaged in a GitHub Release instead of committed to normal Git history. The repository will include the small inference configuration, labels, a model manifest, and a download helper. Quick Start will show both automatic download and manual placement.

The release bundle will preserve the OpenVINO IR files required for the primary demo. Paddle export files may be included in a separate archive when useful, but the default download will remain limited to CPU inference assets.

## Inference behavior

### Image and directory inference

The image command will accept a single file or directory, create output directories automatically, write annotated images, and optionally emit the competition-compatible JSON files. Paths will resolve from command arguments rather than the process working directory.

### Webcam inference

The webcam command will keep asynchronous OpenVINO inference, accept camera index, confidence threshold, IoU threshold, device, and window-size options, and fail clearly when a camera cannot be opened or a frame cannot be read.

### Postprocessing

The model exposes 8,400 candidate boxes and 14 score channels, while the supplied label file documents 13 project classes. Before changing the mapping, a model smoke test will inspect actual outputs and determine whether the final channel is an unused/background score or a missing label. The implementation will validate the channel-to-label relationship explicitly and fail with a useful error instead of silently mislabeling detections.

Non-maximum suppression will be implemented without TensorFlow. It will use OpenCV or a small NumPy implementation, reducing installation size and avoiding an unnecessary framework dependency.

## Demo production

The two road-scene images embedded in the supplied competition material will be used as day and night demo inputs. The existing trained model will generate annotated outputs. These outputs will be visually inspected for readable labels and boxes, then combined into a small looping GIF.

No generated or illustrative detections will be presented as model output.

## README design

The README will follow this order:

1. Title, concise subtitle, badges, and hero demo
2. Overview
3. Core Features
4. Demo
5. Quick Start
6. Dataset Preparation
7. Project Structure
8. Results
9. Model and Inference Pipeline
10. Dataset and Acknowledgements
11. Citation and License

The Results section will directly present the measured project values and a comparison graphic. It will not qualify them as merely copied or historical. Hardware context will be stated where the existing project materials identify it, without inventing missing specifications.

## Dataset and attribution

BDD100K will be linked rather than redistributed. The repository will explain the expected input folders and the original conversion from BDD100K JSON annotations to VOC-style training data. It will acknowledge Berkeley DeepDrive, PaddlePaddle/PaddleDetection, the PP-YOLOE authors, and OpenVINO.

Code originating from or adapted from PaddleDetection will retain appropriate attribution. An Apache-2.0 repository license will be used only after confirming compatibility with included source and assets.

## Verification

Verification will cover:

- Unit tests for preprocessing shapes and dtypes, thresholding, NMS behavior, coordinate scaling, and path handling.
- A class-channel validation test covering the model's 14 score channels and 13 documented labels.
- A model smoke test that loads the OpenVINO IR and runs one supplied image.
- Command help and invalid-input behavior.
- Generation and manual inspection of both demo images and the GIF.
- README link, path, command, and repository-structure checks.
- Secret scanning and large-file checks before publishing.
- A clean clone test using the GitHub Release model download flow.

## Publishing sequence

1. Implement and verify the repository locally in the isolated workspace copy.
2. Create the public GitHub repository under `Jiangyou-rhyyy`.
3. Push the verified source and documentation to `main`.
4. Create the model release and upload the OpenVINO bundle.
5. Update any generated release URLs in the README and download helper.
6. Perform a clean-clone smoke test from the published repository.
7. Synchronize the finished source back to the original local project directory after requesting write access.

## Risks and controls

- OpenVINO version incompatibility: test the supplied IR with an available supported runtime and pin a verified dependency range.
- Old dependency conflicts: replace TensorFlow-only NMS and keep the inference requirements separate from historical training dependencies.
- Misleading metrics: reproduce the project comparison exactly and avoid adding unmeasured hardware claims.
- Large Git history: publish model binaries as release assets.
- Dataset licensing and size: never commit BDD100K data.
- Sensitive or obsolete competition information: exclude the instruction DOCX and scan all published text and metadata.
