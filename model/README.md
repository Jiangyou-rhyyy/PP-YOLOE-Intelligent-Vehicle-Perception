# Model Files

The trained model is distributed through the repository's GitHub Releases so normal Git history stays small.

The default OpenVINO bundle contains:

- `model.xml` — OpenVINO IR graph.
- `model.bin` — OpenVINO IR weights.
- `model.mapping` — optional conversion metadata retained with the exported model.

The release also publishes a `SHA-256` checksum file. The download helper verifies the archive before extracting it into this directory.

Small metadata files remain in Git:

- `infer_cfg.yml` — preprocessing and label metadata from the Paddle export.
- `labels.txt` — one display label per line.

Paddle static-model files are retained as a separate optional release archive and are not required for CPU inference.
