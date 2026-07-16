# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
Bucketed weight transfer via ZMQ + IPC (or shared memory fallback).

Not recommended depending on vllm for this file.
"""

import gc
import hashlib
import logging
import os
from multiprocessing import shared_memory
from typing import Callable, TypedDict

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor

from verl.utils.device import get_device_id, get_device_name, get_torch_device

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

WEIGHT_SYNC_CHECKSUM_ENV = "VERL_WEIGHT_SYNC_CHECKSUM"


class TensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int
    handle: tuple


# copy from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_utils.py
def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer


def create_shared_memory(size: int, name: str):
    """Create shared memory for weight transfer. If already exists, attach to it."""
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        assert shm.size >= size, f"Stale shm segment '{name}': expected {size} bytes, got {shm.size}"
    return shm


def rebuild_shared_memory(name: str, size: int, dtype=torch.uint8):
    """Rebuild tensor from shared memory."""
    shm = shared_memory.SharedMemory(name=name)
    tensor = torch.frombuffer(shm.buf[:size], dtype=dtype)

    return tensor, shm


def update_weight_sync_checksum(checksum, bucket_meta: dict[str, TensorMetadata], shm) -> tuple[int, int]:
    """Hash tensor metadata and bytes from one shared-memory bucket."""
    tensor_count = 0
    byte_count = 0
    for name, meta in bucket_meta.items():
        if meta["handle"] is not None:
            raise ValueError("Weight sync checksum does not support direct IPC tensor handles")
        size = meta["dtype"].itemsize * meta["shape"].numel()
        descriptor = f"{name}\0{tuple(meta['shape'])}\0{meta['dtype']}\0{size}\0".encode()
        checksum.update(descriptor)
        view = shm.buf[meta["offset"] : meta["offset"] + size]
        try:
            checksum.update(view)
        finally:
            view.release()
        tensor_count += 1
        byte_count += size
    return tensor_count, byte_count


class BucketedWeightSender:
    """
    Send model weights via bucketed IPC transfer over ZMQ.

    Packs weight tensors into a fixed-size communication buffer and sends them
    in buckets to the receiver. Supports CUDA IPC and shared memory fallback.

    Args:
        zmq_handle: ZMQ IPC socket path (e.g., "ipc:///tmp/rl-colocate-zmq-<uuid>.sock")
        bucket_size_mb: Communication buffer size in MB
        use_shm: Use shared memory instead of CUDA IPC (for NPU compatibility)
    """

    def __init__(
        self,
        zmq_handle: str,
        bucket_size_mb: int = 512,
        use_shm: bool = False,
        enable_checksum: bool = False,
    ):
        self.zmq_handle = zmq_handle
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size = int(bucket_size_mb) << 20
        self.use_shm = use_shm
        self.enable_checksum = enable_checksum and os.getenv(WEIGHT_SYNC_CHECKSUM_ENV, "0") == "1"
        if self.enable_checksum and not self.use_shm:
            raise ValueError("Weight sync checksum is only supported for shared-memory transfers")

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None
        self._checksum = None
        self._checksum_tensor_count = 0
        self._checksum_byte_count = 0

    async def async_send_weights(self, weights):
        """
        Send weights to the receiver. Accepts a sync generator or async iterator.

        Args:
            weights: Generator or async iterator yielding (name, tensor) pairs
        """
        from verl.workers.rollout.utils import ensure_async_iterator

        try:
            self._init_socket()
            self._init_buffer()
            if self.enable_checksum:
                self._checksum = hashlib.sha256()

            # send bucket weights
            offset = 0
            bucket_meta: dict[str, TensorMetadata] = {}
            # dtype = PrecisionType.to_dtype(self.config.dtype)
            async for name, weight in ensure_async_iterator(weights):
                # model parameters are in fp32 full precision
                # (vermouth1992) we should not force cast weight here because some parameters
                # (such as moe gate) have to keep fp32 precision. If a weight is bf16 in the rollout side,
                # the rollout should automatically cast on demand. However, this would incur a higher weight
                # transfer volume.
                # weight = weight.to(dtype, non_blocking=True)

                # fill the tensor bucket
                if offset + weight.nbytes > self.bucket_size and len(bucket_meta) > 0:
                    get_torch_device().synchronize()
                    self._update_checksum(bucket_meta)
                    self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                    self.socket.recv()
                    bucket_meta = {}
                    offset = 0

                if offset + weight.nbytes > self.bucket_size:
                    assert not self.use_shm, (
                        f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
                        f"Please increase rollout.update_weights_bucket_megabytes({self.bucket_size_mb} MB)."
                    )
                    self._direct_send_large_weight(name, weight)
                    continue

                bucket_meta[name] = {
                    "name": name,
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": offset,
                    "handle": None,
                }
                self.buffer[offset : offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
                offset += weight.nbytes

            # send the last bucket
            get_torch_device().synchronize()
            self._update_checksum(bucket_meta)
            final_metadata = {"bucket_meta": bucket_meta, "is_last": True}
            if self._checksum is not None:
                final_metadata["sender_checksum"] = self._checksum.hexdigest()
                final_metadata["sender_tensor_count"] = self._checksum_tensor_count
                final_metadata["sender_byte_count"] = self._checksum_byte_count
            self.socket.send_pyobj(final_metadata)
            self.socket.recv()
            self._log_checksum("sender")
        finally:
            self._cleanup()

    def _update_checksum(self, bucket_meta: dict[str, TensorMetadata]):
        if self._checksum is None:
            return
        tensor_count, byte_count = update_weight_sync_checksum(self._checksum, bucket_meta, self.shm)
        self._checksum_tensor_count += tensor_count
        self._checksum_byte_count += byte_count

    def _log_checksum(self, role: str):
        if self._checksum is not None:
            logger.warning(
                "Weight sync %s checksum: sha256=%s tensors=%d bytes=%d",
                role,
                self._checksum.hexdigest(),
                self._checksum_tensor_count,
                self._checksum_byte_count,
            )

    def _init_socket(self):
        """Initialize ZMQ REQ socket and bind."""
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.bind(self.zmq_handle)

    def _init_buffer(self):
        """build communication buffer"""
        buffer, shm = None, None
        if not self.use_shm:
            buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device=f"{get_device_name()}:{get_device_id()}")
            handle = reduce_tensor(buffer)
            self.socket.send_pyobj(handle)
        else:
            import uuid

            # Create unique name for shared memory
            shm_name = f"verl_weights_{uuid.uuid4().hex}"
            shm = create_shared_memory(self.bucket_size, shm_name)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)

            comm_metadata = {"name": shm_name, "size": self.bucket_size}
            self.socket.send_pyobj(comm_metadata)

        self.socket.recv()
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        """clean up"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            del self.shm
            self.shm = None
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()

    def _direct_send_large_weight(self, name: str, weight: torch.Tensor):
        """Send a weight larger than the bucket size via cuda ipc or share memory."""
        logger.debug(f"Direct sending large weight {name}({weight.shape}, {weight.dtype})")
        # TODO: support fallback to shared memory
        handle = reduce_tensor(weight)
        bucket_meta: dict[str, TensorMetadata] = {}
        bucket_meta[name] = {
            "name": name,
            "shape": weight.shape,
            "dtype": weight.dtype,
            "offset": 0,
            "handle": handle,
        }
        self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
        self.socket.recv()


class BucketedWeightReceiver:
    """
    Receive model weights via bucketed IPC transfer over ZMQ.

    Receives weight tensors from BucketedWeightSender and passes each
    bucket to a callback for processing (e.g., loading into the model).

    Args:
        zmq_handle: ZMQ IPC socket path (must match sender)
        device: Target device for received tensors
        use_shm: Use shared memory instead of CUDA IPC
    """

    def __init__(
        self,
        zmq_handle: str,
        device: torch.device,
        use_shm: bool = False,
        enable_checksum: bool = False,
    ):
        self.zmq_handle = zmq_handle
        self.device = device
        self.use_shm = use_shm
        self.enable_checksum = enable_checksum and os.getenv(WEIGHT_SYNC_CHECKSUM_ENV, "0") == "1"
        if self.enable_checksum and not self.use_shm:
            raise ValueError("Weight sync checksum is only supported for shared-memory transfers")

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None
        self._checksum = None
        self._checksum_tensor_count = 0
        self._checksum_byte_count = 0

    def receive_weights(self, on_bucket_received: callable):
        """
        Receive weights from sender and process each bucket via callback.

        Args:
            on_bucket_received: Callback function(weights: list[(name, tensor)]) called per bucket.
        """
        try:
            self._init_socket()
            self._init_buffer()
            if self.enable_checksum:
                self._checksum = hashlib.sha256()

            # receive bucket and update weights
            while True:
                metadata = self.socket.recv_pyobj()
                self._update_checksum(metadata["bucket_meta"])
                weights, tensor = [], None
                for name, meta in metadata["bucket_meta"].items():
                    shape, dtype, offset, handle = meta["shape"], meta["dtype"], meta["offset"], meta["handle"]
                    if handle is not None:
                        tensor = rebuild_ipc(handle, self.device.index)
                        weights.append((name, tensor))
                        continue
                    size = dtype.itemsize * shape.numel()
                    tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)
                    if self.use_shm:
                        tensor = tensor.to(self.device)
                    weights.append((name, tensor))
                on_bucket_received(weights)
                get_torch_device().synchronize()
                self.socket.send(b"")
                del weights, tensor
                if metadata["is_last"]:
                    self._verify_sender_checksum(metadata)
                    break
            self._log_checksum("receiver")
        finally:
            self._cleanup()

    def _verify_sender_checksum(self, metadata: dict):
        if self._checksum is None:
            return
        sender_checksum = metadata.get("sender_checksum")
        receiver_checksum = self._checksum.hexdigest()
        if sender_checksum is None:
            raise RuntimeError("Weight sync sender did not provide a checksum")
        if sender_checksum != receiver_checksum:
            raise RuntimeError(
                f"Weight sync checksum mismatch: sender={sender_checksum}, receiver={receiver_checksum}"
            )
        if metadata.get("sender_tensor_count") != self._checksum_tensor_count:
            raise RuntimeError("Weight sync tensor count mismatch")
        if metadata.get("sender_byte_count") != self._checksum_byte_count:
            raise RuntimeError("Weight sync byte count mismatch")
        logger.warning(
            "Weight sync checksum verified: sha256=%s tensors=%d bytes=%d",
            receiver_checksum,
            self._checksum_tensor_count,
            self._checksum_byte_count,
        )

    def _update_checksum(self, bucket_meta: dict[str, TensorMetadata]):
        if self._checksum is None:
            return
        tensor_count, byte_count = update_weight_sync_checksum(self._checksum, bucket_meta, self.shm)
        self._checksum_tensor_count += tensor_count
        self._checksum_byte_count += byte_count

    def _log_checksum(self, role: str):
        if self._checksum is not None:
            logger.warning(
                "Weight sync %s checksum: sha256=%s tensors=%d bytes=%d",
                role,
                self._checksum.hexdigest(),
                self._checksum_tensor_count,
                self._checksum_byte_count,
            )

    def _init_socket(self):
        """Initialize ZMQ REP socket and connect."""
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.connect(self.zmq_handle)

    def _init_buffer(self):
        """Receive and rebuild communication buffer from sender."""
        comm_metadata = self.socket.recv_pyobj()
        buffer, shm = None, None
        if not self.use_shm:
            handle = comm_metadata
            buffer = rebuild_ipc(handle, self.device.index)
            assert buffer.dtype == torch.uint8
        else:
            shm_name = comm_metadata["name"]
            shm_size = comm_metadata["size"]
            buffer, shm = rebuild_shared_memory(shm_name, shm_size, dtype=torch.uint8)
        self.socket.send(b"")
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        """clean up"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        # Synchronize before releasing the buffer to ensure all async ops
        # referencing it (e.g. clone, .to()) have completed.
        get_torch_device().synchronize()
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            del self.shm
            self.shm = None
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()
