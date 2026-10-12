# Model Files

The trained model is distributed through the repository's GitHub Releases so normal Git history stays small.

The `v1.0.0-models` release asset `ppyoloe-openvino-model.zip` contains:

- `model.xml` — OpenVINO IR graph.
- `model.bin` — OpenVINO IR weights.
- `model.mapping` — optional conversion metadata retained with the exported model.

The release also publishes `ppyoloe-openvino-model.zip.sha256` with a `SHA-256`
digest. The download helper verifies the archive before extracting it into this
directory.

Small metadata files remain in Git:

- `infer_cfg.yml` — preprocessing and label metadata from the Paddle export.
- `labels.txt` — one display label per line.

Paddle static-model files are retained as a separate optional release archive and are not required for CPU inference.

## Verified runtime contract

The supplied OpenVINO IR was smoke-tested on CPU with OpenVINO `2024.6.0`, NumPy
`2.1.3`, and OpenCV `4.12.0.88`. Its runtime contract is:

- Input `image`: `[batch, 3, 640, 640]` FP32, with a dynamic batch dimension.
- Boxes `tmp_16`: `[batch, 8400, 4]`.
- Scores `concat_14.tmp_0`: `[batch, 14, 8400]`.

`labels.txt` contains 13 named classes, while the exported model exposes 14 score
channels. On the supplied day and night scenes, channel 13 reached maximum scores
of `0.016435` and `0.009360`, respectively, and never produced a detection at the
default `0.50` confidence threshold. The conversion metadata does not prove that
this channel is background, so the runtime does not silently discard it:

- The reusable detector defaults to `extra_channel_policy="error"`.
- The verified demo workflow uses `extra_channel_policy="label_unknown"`.
- Only the 13 entries in `labels.txt` are presented as named project classes.

Outputs are selected by shape instead of positional order, so conversion-tool
changes that reorder output ports do not swap boxes and scores.
