# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import os
from pathlib import Path
from typing import Optional, Union, Any

import numpy as np
from huggingface_hub import HfApi, HfFolder  # type: ignore
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE  # type: ignore

from infinity_emb._optional_imports import CHECK_ONNXRUNTIME, CHECK_OPTIMUM_AMD

from infinity_emb.log_handler import logger
from infinity_emb.primitives import Device

if CHECK_ONNXRUNTIME.is_available:
    try:
        import onnxruntime as ort  # type: ignore
        from optimum.modeling_base import OptimizedModel  # type: ignore
        from optimum.onnxruntime import (  # type: ignore
            ORTModel,
            ORTOptimizer,
        )
        from optimum.onnxruntime.configuration import OptimizationConfig  # type: ignore
    except (ImportError, RuntimeError, Exception) as ex:
        CHECK_ONNXRUNTIME.mark_dirty(ex)


def mean_pooling(last_hidden_states: np.ndarray, attention_mask: np.ndarray):
    input_mask_expanded = (np.expand_dims(attention_mask, axis=-1)).astype(float)

    sum_embeddings = np.sum(last_hidden_states.astype(float) * input_mask_expanded, axis=1)
    mask_sum = np.maximum(np.sum(input_mask_expanded, axis=1), 1e-9)

    return sum_embeddings / mask_sum


def cls_token_pooling(model_output, *args):
    return model_output[:, 0]


def normalize(input_array, p=2, dim=1, eps=1e-12):
    # Calculate the Lp norm along the specified dimension
    norm = np.linalg.norm(input_array, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    normalized_array = input_array / norm
    return normalized_array


def prepare_ort_inputs(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Ensure inputs match ORT expectations and supply missing position_ids.

    Behavior:
    - Cast input_ids and attention_mask to int64
    - If position_ids missing, create it as int64 using:
        - With attention_mask: cumsum(-1) - 1, then clip at 0
        - Without attention_mask: arange(seq_len) broadcast to batch
    - Leaves any other keys unchanged

    Returns a shallow-copied dict you can pass directly to the ORT model.
    """
    out: dict[str, np.ndarray] = dict(features)

    input_ids = out.get("input_ids")
    attention_mask = out.get("attention_mask")
    position_ids = out.get("position_ids")

    if input_ids is not None:
        input_ids = np.asarray(input_ids, dtype=np.int64)
        out["input_ids"] = input_ids

    if attention_mask is not None:
        attention_mask = np.asarray(attention_mask, dtype=np.int64)
        out["attention_mask"] = attention_mask

    # Only (re)generate when missing
    if position_ids is None and input_ids is not None:
        if attention_mask is not None:
            # position_ids = attention_mask.cumsum(-1) - 1; clip at 0
            position_ids = attention_mask.cumsum(axis=-1) - 1
            position_ids = np.clip(position_ids, 0, None).astype(np.int64, copy=False)
        else:
            # Fallback to simple range per sequence
            if input_ids.ndim != 2:
                raise ValueError(
                    f"Expected input_ids to be 2D (batch, seq_len), got shape {input_ids.shape}"
                )
            batch, seq_len = input_ids.shape
            base = np.arange(seq_len, dtype=np.int64)[None, :]
            # Repeat along batch dimension without tiling data unnecessarily
            position_ids = np.broadcast_to(base, (batch, seq_len)).copy()

        out["position_ids"] = position_ids

    # Ensure dtype for existing position_ids if it was provided
    elif position_ids is not None:
        out["position_ids"] = np.asarray(position_ids, dtype=np.int64)

    return out


def device_to_onnx(device: Device) -> str:
    CHECK_ONNXRUNTIME.mark_required()
    available = ort.get_available_providers()

    if device == Device.cpu:
        if "OpenVINOExecutionProvider" in available:
            return "OpenVINOExecutionProvider"
        return "CPUExecutionProvider"
    elif device == Device.cuda:
        if "ROCMExecutionProvider" in available:
            return "ROCMExecutionProvider"
        elif "MIGraphXExecutionProvider" in available:
            return "MIGraphXExecutionProvider"
        elif "CUDAExecutionProvider" in available:
            return "CUDAExecutionProvider"
        else:
            raise ValueError("No CUDA execution provider available")
    elif device == Device.mps:
        return "CoreMLExecutionProvider"
    elif device == Device.tensorrt:
        # Prefer the new RTX EP if available, otherwise legacy TensorRT, then CUDA/CPU
        if "NvTensorRTRTXExecutionProvider" in available:
            return "NvTensorRTRTXExecutionProvider"
        elif "TensorrtExecutionProvider" in available:
            return "TensorrtExecutionProvider"
        elif "CUDAExecutionProvider" in available:
            return "CUDAExecutionProvider"
        else:
            raise ValueError("No TensorRT execution provider available")
    elif device is None or device == Device.auto:
        # Priority: NvTensorRTRTXExecutionProvider -> CUDAExecutionProvider -> legacy TensorRT -> others -> CPU
        if "NvTensorRTRTXExecutionProvider" in available:
            return "NvTensorRTRTXExecutionProvider"
        elif "TensorrtExecutionProvider" in available:
            return "TensorrtExecutionProvider"
        elif "CUDAExecutionProvider" in available:
            return "CUDAExecutionProvider"
        elif "MIGraphXExecutionProvider" in available:
            return "MIGraphXExecutionProvider"  # swapped order of ROCM and MIGraphX
        elif "ROCMExecutionProvider" in available:
            return "ROCMExecutionProvider"
        elif "CoreMLExecutionProvider" in available:
            return "CoreMLExecutionProvider"
        elif "OpenVINOExecutionProvider" in available:
            return "OpenVINOExecutionProvider"
        else:
            return "CPUExecutionProvider"
    else:
        raise ValueError(f"Unknown device {device}")


def optimize_model(
    model_name_or_path: Union[str, Path],
    model_class: "ORTModel",
    execution_provider: str,
    file_name: str,
    optimize_model=False,
    revision: Optional[str] = None,
    trust_remote_code: bool = True,
    provider_options: Optional[dict[str, Any]] = None,
) -> "OptimizedModel":
    """
    Optimizes, and then loads the model to work best with the execution provider.

    Args:
        model_name_or_path (Union[str, Path]): The model name or path
        model_class (ORTModel): The model class to use, e.g. ORTModelForSequenceClassification
        execution_provider (str): The execution provider to use, e.g. "CUDAExecutionProvider"
        file_name (str): The onnx file name to use, e.g. "model.onnx"
        optimize_model (bool, optional): Whether to optimize the model. Defaults to False.
        revision (Optional[str], optional): The revision to use. Defaults to None.
        trust_remote_code (bool, optional): Whether to trust the remote code. Defaults to True.
    """

    ## If there is no need for optimization
    if execution_provider in {"TensorrtExecutionProvider", "NvTensorRTRTXExecutionProvider"}:
        _kwargs = dict(
            model_id=model_name_or_path,  # Might require model_name_or_path=model_name_or_path for some versions
            revision=revision,
            trust_remote_code=trust_remote_code,
            provider=execution_provider,
            file_name=file_name,
        )
        # Default provider options, caller can override/extend via provider_options
        if execution_provider == "TensorrtExecutionProvider":
            
            base_opts: dict[str, Any] = {
                "trt_fp16_enable": True,
                "trt_layer_norm_fp32_fallback": True,
                "trt_cuda_graph_enable": True,  # helps small layers
                "trt_builder_optimization_level": 3,  # select between 3-5
                # Enhanced engine caching
                "trt_engine_cache_enable": True,
                # Additional optimizations for faster startup
                "trt_timing_cache_enable": True,
                "trt_engine_cache_path": "cache/engines",
                "trt_timing_cache_path": "cache/timing",
                "trt_cache_prefix": "infinity_trt_",
                "trt_weight_stripped_engine_enable": True,
            }
        else:
            # NvTensorRTRTXExecutionProvider
            # Note: paths must be relative for security (absolute paths are rejected)
            base_opts = {
                "enable_cuda_graph": True,
                # Engine caching for RTX provider - must use relative paths
                "engine_cache_path": "cache/engines",
                "engine_cache_prefix": "infinity_trt-rtx_",
                # Timing cache for faster builds - must use relative paths
                "timing_cache_path": "cache/timing",
                "trt_weight_stripped_engine_enable": True,
            }
        if provider_options:
            base_opts.update(provider_options)
        _kwargs["provider_options"] = base_opts
        return model_class.from_pretrained(**_kwargs)

    elif execution_provider in ["ROCMExecutionProvider", "MIGraphXExecutionProvider"]:
        CHECK_OPTIMUM_AMD.mark_required()
        return model_class.from_pretrained(
            model_name_or_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            provider=execution_provider,
            file_name=file_name,
        )

    ## path to find if model has been optimized
    CHECK_ONNXRUNTIME.mark_required()
    path_folder = (
        Path(HUGGINGFACE_HUB_CACHE) / "infinity_onnx" / execution_provider / model_name_or_path
    )
    OPTIMIZED_SUFFIX = "_optimized.onnx"
    files_optimized = list(path_folder.glob(f"**/*{OPTIMIZED_SUFFIX}"))

    logger.info(f"files_optimized: {files_optimized}")
    if files_optimized:
        file_optimized = files_optimized[-1]
        logger.info(f"Optimized model found at {file_optimized}, skipping optimization")
        return model_class.from_pretrained(
            file_optimized.parent.as_posix(),
            revision=revision,
            trust_remote_code=trust_remote_code,
            provider=execution_provider,
            file_name=file_optimized.name,
        )

    unoptimized_model = model_class.from_pretrained(
        model_name_or_path,
        revision=revision,
        trust_remote_code=trust_remote_code,
        provider=execution_provider,
        file_name=file_name,
    )
    if not optimize_model or execution_provider in {"TensorrtExecutionProvider", "NvTensorRTRTXExecutionProvider"}:
        return unoptimized_model
    try:
        logger.info("Optimizing model")

        optimizer = ORTOptimizer.from_pretrained(unoptimized_model)

        is_gpu = not (
            "cpu" in execution_provider.lower() or "openvino" in execution_provider.lower()
        )
        optimization_config = OptimizationConfig(
            optimization_level=99,
            optimize_with_onnxruntime_only=False,
            optimize_for_gpu=is_gpu,
            fp16=is_gpu,
            # enable_gelu_approximation=True,
            # enable_gemm_fast_gelu_fusion=True, # might not work
        )

        optimized_model_path = optimizer.optimize(
            optimization_config=optimization_config,
            save_dir=path_folder.as_posix(),
            # if larger than 2gb use external data format
            one_external_file=True,
        )

        model = model_class.from_pretrained(
            optimized_model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            provider=execution_provider,
            file_name=Path(file_name).name.replace(".onnx", OPTIMIZED_SUFFIX),
        )
    except Exception as e:
        logger.warning(f"Optimization failed with {e}. Going to use the unoptimized model.")
        model = unoptimized_model

    return model


def _list_all_repo_files(
    model_name_or_path: str,
    revision: Union[str, None] = None,
    use_auth_token: Union[bool, str] = True,
):
    if not Path(model_name_or_path).exists():
        if isinstance(use_auth_token, bool):
            token = HfFolder().get_token()
        else:
            token = use_auth_token
        return list(
            map(
                Path,
                HfApi().list_repo_files(model_name_or_path, revision=revision, token=token),
            )
        )
    else:
        return list(Path(model_name_or_path).glob("**/*"))


def get_onnx_files(
    *,
    model_name_or_path: str,
    revision: Union[str, None] = None,
    use_auth_token: Union[bool, str] = True,
    prefer_quantized=False,
) -> Path:
    """gets the onnx files from the repo"""
    repo_files = _list_all_repo_files(
        model_name_or_path=model_name_or_path,
        revision=revision,
        use_auth_token=use_auth_token,
    )
    pattern = "**.onnx"
    onnx_files = [p for p in repo_files if p.match(pattern)]

    prefered_regex = "quantize" if prefer_quantized else "model.onnx"
    prefered_onnx = [f for f in onnx_files if prefered_regex in f.name]
    if len(onnx_files) > 1:
        logger.info(f"Found {len(onnx_files)} onnx files: {onnx_files}")
        if prefered_onnx:
            onnx_files = prefered_onnx
        onnx_file = onnx_files[-1]
        logger.info(f"Using {onnx_file} as the model")
        return onnx_file
    elif len(onnx_files) == 1:
        return onnx_files[0]
    else:
        raise ValueError(f"No onnx files found for {model_name_or_path} and revision {revision}")
