# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import copy
import os

import numpy as np
from pathlib import Path
from typing import Optional

from infinity_emb._optional_imports import CHECK_ONNXRUNTIME, CHECK_TRANSFORMERS
from infinity_emb.args import EngineArgs
from infinity_emb.primitives import EmbeddingReturnType, PoolingMethod
from infinity_emb.transformer.abstract import BaseEmbedder
from infinity_emb.transformer.quantization.interface import quant_embedding_decorator
from infinity_emb.transformer.utils_optimum import (
    cls_token_pooling,
    device_to_onnx,
    get_onnx_files,
    mean_pooling,
    normalize,
    optimize_model,
    prepare_ort_inputs,
)
from infinity_emb.log_handler import logger

if CHECK_ONNXRUNTIME.is_available:
    try:
        from optimum.onnxruntime import (  # type: ignore[import-untyped]
            ORTModelForFeatureExtraction,
        )

    except (ImportError, RuntimeError, Exception) as ex:
        CHECK_ONNXRUNTIME.mark_dirty(ex)

if CHECK_TRANSFORMERS.is_available:
    from transformers import AutoConfig, AutoTokenizer  # type: ignore[import-untyped]


class OptimumEmbedder(BaseEmbedder):
    def __init__(self, *, engine_args: EngineArgs):
        CHECK_ONNXRUNTIME.mark_required()
        provider = device_to_onnx(engine_args.device)

        # Prepare tokenizer/config first so we can shape TensorRT dynamic profiles
        self.tokenizer = AutoTokenizer.from_pretrained(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
        )
        self.config = AutoConfig.from_pretrained(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
        )
        self._infinity_tokenizer = copy.deepcopy(self.tokenizer)
        self.engine_args = engine_args

        onnx_file = get_onnx_files(
            model_name_or_path=engine_args.model_name_or_path,
            revision=engine_args.revision,
            use_auth_token=True,
            prefer_quantized=("cpu" in provider.lower() or "openvino" in provider.lower()),
        )
        # Optionally patch ONNX so position_ids is declared as INT64 (needed by TensorRT)
        onnx_file = self._maybe_patch_position_ids_to_int64(onnx_file)

        # If we have a local (possibly patched) ONNX file path, prefer loading from its directory
        if onnx_file.is_absolute() or onnx_file.exists():
            model_id_for_load = onnx_file.parent.as_posix()
            file_name_for_load = onnx_file.name
        else:
            # Fall back to repo id + repo-relative path
            model_id_for_load = engine_args.model_name_or_path
            file_name_for_load = onnx_file.as_posix()

        logger.info(f"[infinity] ONNX load path: dir={model_id_for_load} file={file_name_for_load}")

        self.pooling = (
            mean_pooling if engine_args.pooling_method == PoolingMethod.mean else cls_token_pooling
        )

        provider_options = None
        if "tensorrt" in provider.lower():
            # Define dynamic shape profiles for TensorRT to avoid 0/32767 defaults
            max_seq = int(os.getenv("INFINITY_TRT_MAX_SEQ_LEN", self.config.max_position_embeddings))
            max_bs = int(os.getenv("INFINITY_TRT_MAX_BATCH", 16))
            opt_seq = int(os.getenv("INFINITY_TRT_OPT_SEQ_LEN", min(256, max_seq)))
            opt_bs = int(os.getenv("INFINITY_TRT_OPT_BATCH", min(8, max_bs)))
            min_seq = int(os.getenv("INFINITY_TRT_MIN_SEQ_LEN", 1))
            min_bs = int(os.getenv("INFINITY_TRT_MIN_BATCH", 1))

            def shape_str(bs: int, seqlen: int) -> str:
                return ",".join(
                    [
                        f"input_ids:{bs}x{seqlen}",
                        f"attention_mask:{bs}x{seqlen}",
                        f"position_ids:{bs}x{seqlen}",
                    ]
                )

            min_shapes = shape_str(min_bs, min_seq)
            opt_shapes = shape_str(opt_bs, opt_seq)
            max_shapes = shape_str(max_bs, max_seq)

            if provider == "NvTensorRTRTXExecutionProvider":
                provider_options = {
                    "nv_profile_min_shapes": min_shapes,
                    "nv_profile_opt_shapes": opt_shapes,
                    "nv_profile_max_shapes": max_shapes,
                    "enable_cuda_graph": True,
                }
            else:
                provider_options = {
                    "trt_profile_min_shapes": min_shapes,
                    "trt_profile_opt_shapes": opt_shapes,
                    "trt_profile_max_shapes": max_shapes,
                }

        self.model = optimize_model(
            model_name_or_path=model_id_for_load,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
            execution_provider=provider,
            file_name=file_name_for_load,
            optimize_model=not os.environ.get(
                "INFINITY_ONNX_DISABLE_OPTIMIZE", False
            ),  # TODO: make this env variable public
            model_class=ORTModelForFeatureExtraction,
            provider_options=provider_options,
        )
        self.model.use_io_binding = False

        # Cache ONNX input names to avoid passing unexpected feeds
        self._input_names = set()
        try:
            session = getattr(self.model, "model", None)
            if session is not None and hasattr(session, "get_inputs"):
                self._input_names = {i.name for i in session.get_inputs()}
                # One-time visibility: print input dtypes for troubleshooting
                try:
                    dtypes = {i.name: i.type for i in session.get_inputs()}
                    logger.info(f"[infinity] ONNX inputs: {dtypes}")
                except Exception:
                    pass
        except Exception:
            # Best-effort; we'll lazily refresh in encode_core if needed
            self._input_names = set()

    def _maybe_patch_position_ids_to_int64(self, onnx_path: Path) -> Path:
        """If ONNX declares input_ids, attention_mask, or position_ids not as INT64, optionally patch them.
        Enable via env INFINITY_PATCH_ONNX_POSITION_IDS=1/true/yes.
        """
        try:
            import onnx  # type: ignore
            from onnx import TensorProto  # type: ignore
        except Exception:
            return onnx_path

        try:
            # Ensure we have a local file; if not, snapshot the repo so external data exists
            local_path = onnx_path
            snapshot_dir: Optional[Path] = None
            if not local_path.exists():
                try:
                    from huggingface_hub import snapshot_download  # type: ignore

                    snapshot_dir = Path(
                        snapshot_download(
                            repo_id=self.engine_args.model_name_or_path,
                            revision=self.engine_args.revision,
                            allow_patterns=["*.onnx", "*.onnx_data", "**/*.onnx", "**/*.onnx_data"],
                        )
                    )
                    # Try exact relative path first
                    candidate = snapshot_dir / onnx_path.as_posix()
                    if candidate.exists():
                        local_path = candidate
                    else:
                        # Fallback: search by filename within snapshot
                        matches = list(snapshot_dir.rglob(onnx_path.name))
                        if not matches:
                            logger.warning(
                                f"[infinity] Could not inspect/patch ONNX ({onnx_path.name}): not found in snapshot"
                            )
                            return onnx_path
                        local_path = matches[-1]
                except Exception:
                    # Couldn't resolve locally; skip patching
                    logger.warning(
                        f"[infinity] Could not inspect/patch ONNX ({onnx_path.name}): not found locally and download failed"
                    )
                    return onnx_path

            if snapshot_dir is not None:
                logger.info(f"[infinity] ONNX snapshot dir: {snapshot_dir}")

            logger.info(f"[infinity] Inspecting ONNX file for INT64 bindings: {local_path}")

            # Load model with external data when present
            try:
                model = onnx.load_model(local_path.as_posix(), load_external_data=True)
            except Exception:
                model = onnx.load(local_path.as_posix())
            
            # Check all three critical inputs that need INT64 binding
            inputs_to_patch = ["input_ids", "attention_mask", "position_ids"]
            patched_inputs = []
            inputs_found = {}
            
            for vi in model.graph.input:
                if vi.name in inputs_to_patch:
                    inputs_found[vi.name] = vi
                    elem = vi.type.tensor_type.elem_type
                    logger.info(f"[infinity] Found input {vi.name} with type {elem} (INT64={TensorProto.INT64})")
                    if elem != TensorProto.INT64:
                        patched_inputs.append(vi.name)

            logger.info(f"[infinity] Inputs needing INT64 patching: {patched_inputs}")

            if not patched_inputs:
                logger.info(f"[infinity] No inputs need patching, all are already INT64")
                return local_path

            do_patch = os.getenv("INFINITY_PATCH_ONNX_POSITION_IDS", "0").lower() in ("1", "true", "yes")
            logger.info(f"[infinity] INFINITY_PATCH_ONNX_POSITION_IDS={os.getenv('INFINITY_PATCH_ONNX_POSITION_IDS', '0')}, do_patch={do_patch}")
            if not do_patch:
                print(
                    f"[infinity] WARNING: ONNX inputs {patched_inputs} are not INT64. "
                    "TensorRT/ORT may warn about input binding types. Set INFINITY_PATCH_ONNX_POSITION_IDS=1 to write a patched copy, "
                    "or re-export the model with INT64 input types for input_ids, attention_mask, and position_ids."
                )
                return local_path

            # Check if patched file already exists
            patched = local_path.with_suffix(".int64.onnx")
            if patched.exists():
                logger.info(f"[infinity] Using existing patched ONNX: {patched}")
                return patched

            # Patch input dtypes to INT64 and write to a sibling file
            for input_name in patched_inputs:
                if input_name in inputs_found:
                    inputs_found[input_name].type.tensor_type.elem_type = TensorProto.INT64
                    logger.info(f"[infinity] Patching {input_name} to INT64")
            
            # Save patched model; include external data in a single sidecar file
            try:
                onnx.save_model(
                    model,
                    patched.as_posix(),
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=patched.name + "_data",
                    size_threshold=1024,
                )
            except Exception:
                onnx.save(model, patched.as_posix())
            logger.info(f"[infinity] Patched ONNX saved at: {patched}")
            return patched
        except Exception as e:
            logger.warning(f"[infinity] Could not inspect/patch ONNX ({onnx_path.name}): {e}")
            return onnx_path

    def encode_pre(self, sentences: list[str]) -> dict[str, np.ndarray]:
        encoded = self.tokenizer(
            sentences,
            max_length=self.config.max_position_embeddings,
            padding=True,
            truncation="longest_first",
            return_tensors="np",
            return_token_type_ids=False,
            pad_to_multiple_of=8,
        )

        # Use centralized function to ensure all inputs have proper int64 bindings
        return prepare_ort_inputs(encoded)

    def encode_core(self, onnx_input: dict[str, np.ndarray]) -> dict:
        # Lazily determine allowed input names if not already cached
        if not self._input_names:
            try:
                session = getattr(self.model, "model", None)
                if session is not None and hasattr(session, "get_inputs"):
                    self._input_names = {i.name for i in session.get_inputs()}
            except Exception:
                self._input_names = set(onnx_input.keys())

        filtered = {k: v for k, v in onnx_input.items() if not self._input_names or k in self._input_names}
        
        # Debug: Log what data types we're actually sending
        if logger.isEnabledFor(10):  # DEBUG level
            input_dtypes = {k: str(v.dtype) for k, v in filtered.items()}
            logger.debug(f"[infinity] Sending ONNX inputs with dtypes: {input_dtypes}")

        outputs = self.model(**filtered)
        return {
            "token_embeddings": outputs["last_hidden_state"],
            "attention_mask": filtered["attention_mask"],
        }

    @quant_embedding_decorator()
    def encode_post(self, embedding: dict) -> np.ndarray:
        embedding = self.pooling(  # type: ignore
            embedding["token_embeddings"], embedding["attention_mask"]
        )

        return normalize(embedding).astype(np.float32)

    def tokenize_lengths(self, sentences: list[str]) -> list[int]:
        if hasattr(self._infinity_tokenizer, "encode_batch"):
            tks = self._infinity_tokenizer.encode_batch(
                sentences,
                padding=False,
                truncation="longest_first",
            )
        else:
            tks = self._infinity_tokenizer(sentences, padding=False, truncation="longest_first")

        return [len(t) for t in tks["input_ids"]]
