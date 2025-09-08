# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import copy
import os

from infinity_emb._optional_imports import CHECK_ONNXRUNTIME, CHECK_TRANSFORMERS
from infinity_emb.args import EngineArgs
from infinity_emb.transformer.abstract import BaseClassifer
from infinity_emb.transformer.utils_optimum import (
    device_to_onnx,
    get_onnx_files,
    optimize_model,
)

if CHECK_ONNXRUNTIME.is_available:
    try:
        from optimum.onnxruntime import (  # type: ignore[import-untyped]
            ORTModelForSequenceClassification,
        )

    except (ImportError, RuntimeError, Exception) as ex:
        CHECK_ONNXRUNTIME.mark_dirty(ex)

if CHECK_TRANSFORMERS.is_available:
    from transformers import AutoTokenizer, AutoConfig, pipeline  # type: ignore[import-untyped]


class OptimumClassifier(BaseClassifer):
    def __init__(self, *, engine_args: EngineArgs):
        CHECK_ONNXRUNTIME.mark_required()
        CHECK_TRANSFORMERS.mark_required()
        provider = device_to_onnx(engine_args.device)

        # Prepare tokenizer/config first so we can shape TensorRT dynamic profiles
        self.tokenizer = AutoTokenizer.from_pretrained(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
        )
        self._infinity_tokenizer = copy.deepcopy(self.tokenizer)
        self.config = AutoConfig.from_pretrained(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
        )

        onnx_file = get_onnx_files(
            model_name_or_path=engine_args.model_name_or_path,
            revision=engine_args.revision,
            use_auth_token=True,
            prefer_quantized=("cpu" in provider.lower() or "openvino" in provider.lower()),
            onnx_filename=engine_args.onnx_filename,
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

        model = optimize_model(
            model_name_or_path=engine_args.model_name_or_path,
            model_class=ORTModelForSequenceClassification,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
            execution_provider=provider,
            file_name=onnx_file.as_posix(),
            optimize_model=not os.environ.get("INFINITY_ONNX_DISABLE_OPTIMIZE", False),
            provider_options=provider_options,
        )
        model.use_io_binding = False

        self._pipe = pipeline(
            task="text-classification",
            model=model,
            trust_remote_code=engine_args.trust_remote_code,
            top_k=None,
            revision=engine_args.revision,
            tokenizer=self.tokenizer,
        )

    def encode_pre(self, sentences: list[str]):
        return sentences

    def encode_core(self, sentences: list[str]) -> dict:
        outputs = self._pipe(sentences)
        return outputs

    def encode_post(self, classes) -> dict[str, float]:
        """runs post encoding such as normalization"""
        return classes

    def tokenize_lengths(self, sentences: list[str]) -> list[int]:
        """gets the lengths of each sentences according to tokenize/len etc."""
        tks = self._infinity_tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_length=False,
        ).encodings
        return [len(t.tokens) for t in tks]
