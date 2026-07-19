"""ROCm Accelerator plugin for AudioMuse-AI.

Wires AMD GPU acceleration into the analysis pipeline without forking core:

* musicnn and the CLAP audio encoder run on the AMD GPU through ONNX Runtime's
  MIGraphXExecutionProvider, scoped to those two models only. CLAP needs its
  symbolic time axis pinned to a static shape before MIGraphX can compile it
  (see ``tasks/clap_analyzer.py:_prepared_model_bytes``). The Whisper decoder
  has no such fixup, so it stays off this chain entirely.
* lyrics ASR is swapped to faster-whisper (CTranslate2's ROCm backend) because
  MIGraphX can't run the ONNX Whisper decoder at all.

The ROCm runtime (MIGraphX-enabled onnxruntime, CTranslate2 ROCm, faster-whisper,
GPU device access) is provided by the ROCm worker image, not this plugin. On any
other image the plugin registers nothing and stays inert.
"""

import logging

from plugin.api import get_setting

logger = logging.getLogger("plugin.rocm_accelerator")


def _asr_factory():
    # Imported lazily on the worker so non-ROCm containers never touch it.
    from . import whisper_faster

    return whisper_faster


def _migraphx_available():
    try:
        import onnxruntime as ort

        return "MIGraphXExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _faster_whisper_available():
    try:
        import faster_whisper  # noqa: F401

        return True
    except Exception:
        return False


def register(ctx):
    # Guard: this plugin only does anything on the ROCm worker image. On the CPU
    # or CUDA image the runtime is absent, so register nothing and leave analysis
    # exactly as it was rather than offering a provider that can't load.
    if not _migraphx_available():
        logger.warning(
            "MIGraphXExecutionProvider not available - ROCm Accelerator stays inert. "
            "Install this plugin on the AudioMuse-AI ROCm worker image."
        )
        return

    # fp16 defaults on. A GPU page fault (mul_add_kernel / convert_mul_add_kernel)
    # has been seen on gfx1201 (RX 9070 XT / RDNA4) during MusiCNN/CLAP inference
    # with fp16 both on and off, so it isn't fp16-specific - disabling fp16 buys
    # no safety, just throughput. Opt out via the plugin's settings if needed.
    options = {
        "device_id": 0,
        "migraphx_model_cache_dir": "/app/.cache/migraphx",
    }
    if get_setting("fp16_enable", True):
        options["migraphx_fp16_enable"] = "1"

    ctx.register_onnx_provider(
        "MIGraphXExecutionProvider",
        options,
        only_models=["musicnn", "clap"],
    )
    logger.info(
        "Registered MIGraphX ONNX provider for musicnn and CLAP audio (AMD GPU, fp16=%s)",
        options.get("migraphx_fp16_enable", "0"),
    )

    if _faster_whisper_available():
        ctx.register_analysis_provider("asr", _asr_factory)
        logger.info("Registered faster-whisper as the ASR backend (AMD GPU)")
    else:
        logger.warning(
            "faster_whisper not importable - lyrics ASR stays on the ONNX backend (CPU). "
            "musicnn acceleration is unaffected."
        )
