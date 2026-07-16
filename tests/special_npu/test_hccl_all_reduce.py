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

"""Minimal multi-NPU HCCL smoke test.

Run with:
    torchrun --standalone --nnodes=1 --nproc-per-node=8 \
        tests/special_npu/test_hccl_all_reduce.py
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", timeout=timedelta(seconds=180))

    try:
        value = torch.tensor([rank + 1], dtype=torch.float32, device=f"npu:{local_rank}")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.npu.synchronize()

        expected = world_size * (world_size + 1) / 2
        actual = value.item()
        if actual != expected:
            raise AssertionError(f"rank {rank}: expected {expected}, got {actual}")

        print(
            f"HCCL_RANK_PASS rank={rank} local_rank={local_rank} "
            f"world_size={world_size} value={actual}",
            flush=True,
        )
        dist.barrier()
        if rank == 0:
            print(f"HCCL_SMOKE_PASS world_size={world_size} sum={actual}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
