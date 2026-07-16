# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from functools import wraps

import numpy as np
import torch

OPENPANGU_ARCHITECTURE = "OpenPanguVLForConditionalGeneration"
OPENPANGU_MODEL_TYPE = "openpangu_vl"
OPENPANGU_KEY_MAPPING = {
    "^visual": "model.visual",
    r"^model(?!\.(language_model|visual))": "model.language_model",
}


def _compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
    del seq_len
    config.standardize_rope_params()
    rope_parameters = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
    base = rope_parameters["rope_theta"]
    partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).to(dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


def apply_openpangu_transformers_compat() -> None:
    """Install the minimal compatibility required by OpenPangu remote code."""
    import transformers.utils as transformers_utils

    if not hasattr(transformers_utils, "LossKwargs"):
        transformers_utils.LossKwargs = transformers_utils.TransformersKwargs

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    try:
        from transformers.conversion_mapping import (
            WeightRenaming,
            get_checkpoint_conversion_mapping,
            register_checkpoint_conversion_mapping,
        )
    except ImportError:
        # Transformers 4 applies the model class's legacy mapping through this allowlist.
        from transformers import modeling_utils

        if "openpanguvl" not in modeling_utils.VLMS:
            modeling_utils.VLMS.append("openpanguvl")
        return

    if get_checkpoint_conversion_mapping(OPENPANGU_MODEL_TYPE) is None:
        register_checkpoint_conversion_mapping(
            OPENPANGU_MODEL_TYPE,
            [
                WeightRenaming(source_patterns=source, target_patterns=target)
                for source, target in OPENPANGU_KEY_MAPPING.items()
            ],
        )


def apply_openpangu_config_compat(config) -> None:
    """Fill fields required when the vision submodel initializes weights."""
    vision_config = config.vision_config
    if not hasattr(vision_config, "initializer_range"):
        vision_config.initializer_range = config.text_config.initializer_range

    for token_id_name in ("pad_token_id", "bos_token_id", "eos_token_id"):
        token_id = getattr(config, token_id_name, None)
        if token_id is not None:
            setattr(config.text_config, token_id_name, token_id)


def patch_openpangu_processor(processor) -> None:
    """Provide NumPy missing from the published OpenPangu processor module."""
    processor_module = sys.modules[processor.__class__.__module__]
    if not hasattr(processor_module, "np"):
        processor_module.np = np


def normalize_openpangu_position_ids(position_ids: torch.Tensor | None) -> torch.Tensor | None:
    """Drop Verl's leading text position row from OpenPangu mRoPE positions."""
    if position_ids is not None and position_ids.ndim == 3 and position_ids.shape[0] == 4:
        return position_ids[1:]
    return position_ids


def patch_openpangu_model_position_ids(model) -> None:
    """Make the OpenPangu remote-code forward accept Verl's mRoPE layout."""
    model_class = model.__class__
    if getattr(model_class, "_verl_position_ids_compat", False):
        return

    original_forward = model_class.forward

    @wraps(original_forward)
    def forward(self, *args, position_ids=None, **kwargs):
        return original_forward(
            self,
            *args,
            position_ids=normalize_openpangu_position_ids(position_ids),
            **kwargs,
        )

    model_class.forward = forward
    model_class._verl_position_ids_compat = True


__all__ = [
    "OPENPANGU_ARCHITECTURE",
    "OPENPANGU_KEY_MAPPING",
    "OPENPANGU_MODEL_TYPE",
    "apply_openpangu_config_compat",
    "apply_openpangu_transformers_compat",
    "normalize_openpangu_position_ids",
    "patch_openpangu_processor",
    "patch_openpangu_model_position_ids",
]
