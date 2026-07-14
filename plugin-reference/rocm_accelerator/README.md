# ROCm Accelerator (reference plugin)

Proof-of-concept AudioMuse-AI plugin that adds AMD GPU acceleration to the
analysis pipeline, using only two generic plugin seams so nothing AMD-specific
lives in core.

## What it does

- **musicnn on the AMD GPU** via ONNX Runtime's `MIGraphXExecutionProvider`,
  registered with `only_models=["musicnn"]` so it never touches CLAP or Whisper
  (MIGraphX can't parse those graphs).
- **lyrics ASR on the AMD GPU** by registering `faster_whisper.py` as the `asr`
  analysis provider (`register_analysis_provider('asr', ...)`), replacing the
  built-in ONNX Whisper backend that MIGraphX can't run.

CLAP audio and clustering (RAPIDS cuML) stay on CPU: MIGraphX can't parse CLAP's
Resize op, and cuML has no ROCm port.

## Requirements

Runs only on the **AudioMuse-AI ROCm worker image**, which provides the
MIGraphX-enabled onnxruntime, CTranslate2's ROCm build, faster-whisper and GPU
device access. `requirements` in `plugin.json` is intentionally empty: a PyPI
`onnxruntime` would clobber the image's MIGraphX build. On any other image the
plugin detects the missing provider and stays inert.

## Core seams it depends on

1. `register_onnx_provider(..., only_models=/exclude_models=)` - per-model
   provider scoping (`plugin/api.py`, `tasks/analysis_helper.py`).
2. `register_analysis_provider('asr', factory)` - component replacement, resolved
   by `lyrics/_asr_backend.py` before the built-in `whisper_onnx`.

## Env (set by the ROCm image, override if needed)

- `LYRICS_WHISPER_FASTER_DEVICE` (default `cuda`; CTranslate2 mirrors the CUDA
  API on ROCm)
- `LYRICS_WHISPER_FASTER_COMPUTE_TYPE` (default `float16`)
- `LYRICS_WHISPER_FASTER_MODEL_DIR` (default `/app/model/faster-whisper-small`)
