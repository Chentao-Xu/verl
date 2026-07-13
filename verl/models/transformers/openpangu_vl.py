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

import transformers.utils as transformers_utils
from functools import wraps

import torch
from transformers import modeling_utils


def apply_openpangu_transformers_compat() -> None:
    """Apply compatibility needed by OpenPangu remote code on Transformers 4.57."""
    if not hasattr(transformers_utils, "LossKwargs"):
        from transformers.utils import TransformersKwargs

        transformers_utils.LossKwargs = TransformersKwargs

    # Transformers only applies checkpoint conversion mappings automatically
    # to a fixed list of legacy VLMs. OpenPangu uses the same old-to-new key
    # conversion mechanism but is not included in that list.
    if "openpanguvl" not in modeling_utils.VLMS:
        modeling_utils.VLMS.append("openpanguvl")


def apply_openpangu_config_compat(config) -> None:
    """Fill config fields required by Transformers weight initialization."""
    vision_config = config.vision_config
    if not hasattr(vision_config, "initializer_range"):
        vision_config.initializer_range = config.text_config.initializer_range


def normalize_openpangu_position_ids(position_ids: torch.Tensor | None) -> torch.Tensor | None:
    """Convert Verl's text-plus-mRoPE positions to OpenPangu's 3D mRoPE."""
    if position_ids is not None and position_ids.ndim == 3 and position_ids.shape[0] == 4:
        return position_ids[1:]
    return position_ids


def patch_openpangu_model_position_ids(model) -> None:
    """Make the remote OpenPangu forward accept Verl's position-id layout."""
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


apply_openpangu_transformers_compat()


__all__ = [
    "apply_openpangu_config_compat",
    "apply_openpangu_transformers_compat",
    "normalize_openpangu_position_ids",
    "patch_openpangu_model_position_ids",
]
