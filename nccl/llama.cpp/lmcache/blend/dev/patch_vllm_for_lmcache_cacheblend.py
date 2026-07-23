#!/usr/bin/env python3
from __future__ import annotations

import inspect
from pathlib import Path

import vllm.v1.worker.gpu_worker as gpu_worker


LOAD_MODEL_OLD = """    def load_model(self) -> None:
        eep_scale_up = os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1"
        with self._maybe_get_memory_pool_context(tag="weights"):
            self.model_runner.load_model(eep_scale_up=eep_scale_up)
"""

LOAD_MODEL_NEW = """    def load_model(self) -> None:
        eep_scale_up = os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1"
        with self._maybe_get_memory_pool_context(tag="weights"):
            self.model_runner.load_model(eep_scale_up=eep_scale_up)
            if self.vllm_config.kv_transfer_config is not None:
                try:
                    from lmcache.integration.vllm.utils import ENGINE_NAME
                    from lmcache.v1.compute.models.utils import VLLMModelTracker

                    VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.model)
                    ensure_kv_transfer_initialized(self.vllm_config)
                except Exception:
                    logger.exception("Failed to initialize LMCache CacheBlend after model load")
                    raise
"""

INIT_OLD = """    ensure_kv_transfer_initialized(vllm_config)
"""

INIT_NEW = """    # LMCache CacheBlend needs the vLLM model to be registered before
    # initializing KV transfer. GPUWorker.load_model() performs that ordering.
"""


def main() -> None:
    path = Path(inspect.getfile(gpu_worker))
    text = path.read_text(encoding="utf-8")
    original = text

    if "VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.model)" not in text:
        if LOAD_MODEL_OLD not in text:
            raise RuntimeError(f"Could not find load_model patch point in {path}")
        text = text.replace(LOAD_MODEL_OLD, LOAD_MODEL_NEW)

    if INIT_OLD in text:
        text = text.replace(INIT_OLD, INIT_NEW, 1)

    if text == original:
        print(f"vLLM CacheBlend patch already applied: {path}")
        return

    path.write_text(text, encoding="utf-8")
    print(f"Applied vLLM CacheBlend patch: {path}")


if __name__ == "__main__":
    main()
