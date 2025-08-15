from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from tokenize import group
from typing import List

import numpy as np
import torch
import torch.distributed as dist


def _make_buffer(size_bytes: int, device: str, gpu_idx: int):
    """Allocate a contiguous tensor/array of *size_bytes* and return it."""
    n_elems = size_bytes // 4  # float32
    if device == "gpu":
        tensor = torch.ones(n_elems, dtype=torch.float32, device=f"cuda:{gpu_idx}")
        return tensor
    arr = np.ones(n_elems, dtype=np.float32)
    return arr


def _pretty_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    val = float(num_bytes)
    for u in units:
        if val < 1024 or u == units[-1]:
            return f"{val:.0f} {u}" if u == "B" else f"{val:.1f} {u}"
        val /= 1024
    return f"{num_bytes} B"


def _send(tensor, dst, async_op=False):
    if isinstance(tensor, torch.Tensor):
        if async_op:
            return dist.isend(tensor, dst=dst)
        else:
            dist.send(tensor, dst=dst)
    else:  # numpy array (gloo backend)
        t = torch.from_numpy(tensor)
        if async_op:
            return dist.isend(t, dst=dst)
        else:
            dist.send(t, dst=dst)


def _recv(tensor, src, async_op=False):
    if isinstance(tensor, torch.Tensor):
        if async_op:
            return dist.irecv(tensor, src=src)
        else:
            dist.recv(tensor, src=src)
    else:
        t = torch.from_numpy(tensor)
        if async_op:
            return dist.irecv(t, src=src)
        else:
            dist.recv(t, src=src)
        tensor[:] = t.cpu().numpy()


################################################################################
# Benchmark roles
################################################################################


def _run_server(args):
    peer = 0  # client rank
    for size in args.sizes:
        tensor = _make_buffer(size, args.device, args.local_gpu_idx)
        tensor2 = _make_buffer(size, args.device, args.local_gpu_idx)
        # Warm-up receive
        _recv(tensor, src=peer)
        torch.cuda.synchronize() if isinstance(tensor, torch.Tensor) else None

        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            recv_op = dist.P2POp(dist.irecv, tensor, peer)
            send_op = dist.P2POp(dist.isend, tensor2, peer)
            reqs = dist.batch_isend_irecv([send_op, recv_op])
            for req in reqs:
                req.wait()
            total += size
        torch.cuda.synchronize() if isinstance(tensor, torch.Tensor) else None
        elapsed = time.perf_counter() - start
        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        print(
            f"[Server] {_pretty_size(size):>9} : {gbps:7.2f} Gbps | {gb_sec:7.2f} GB/s"
        )
    print("[Server] Benchmark complete")


def _run_client(args):

    peer = 1  # server rank
    for size in args.sizes:
        tensor = _make_buffer(size, args.device, args.local_gpu_idx)
        tensor2 = _make_buffer(size, args.device, args.local_gpu_idx)
        # Warm-up send
        _send(tensor, dst=peer)
        torch.cuda.synchronize() if isinstance(tensor, torch.Tensor) else None

        start = time.perf_counter()
        total = 0
        for _ in range(args.iters):
            send_op = dist.P2POp(dist.isend, tensor, peer)
            recv_op = dist.P2POp(dist.irecv, tensor2, peer)
            reqs = dist.batch_isend_irecv([send_op, recv_op])
            for req in reqs:
                req.wait()
            total += size
        torch.cuda.synchronize() if isinstance(tensor, torch.Tensor) else None
        elapsed = time.perf_counter() - start
        gbps = (total * 8) / elapsed / 1e9
        gb_sec = total / elapsed / 1e9
        print(
            f"[Client] {_pretty_size(size):>9} : {gbps:7.2f} Gbps | {gb_sec:7.2f} GB/s"
        )
    print("[Client] Benchmark complete")


def parse_size_list(val: str) -> List[int]:
    try:
        return [int(s) for s in val.split(",") if s]
    except ValueError:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers")


def main():
    p = argparse.ArgumentParser(
        description="Benchmark NCCL (torch.distributed) bandwidth"
    )
    p.add_argument("--local-gpu-idx", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    p.add_argument(
        "--sizes",
        type=parse_size_list,
        default=[
            256,
            1024,
            4096,
            16384,
            65536,
            262144,
            1048576,
            10485760,
            104857600,
        ],
    )
    p.add_argument("--iters", type=int, default=1000)
    args = p.parse_args()

    backend = "nccl" if args.device == "gpu" else "gloo"
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This benchmark only supports 2 processes"

    print("NCCL Benchmark — role:", "client" if rank == 0 else "server")
    print("Message sizes:", ", ".join(_pretty_size(s) for s in args.sizes))
    print(
        f"Device: {args.device} | Local GPU idx: {args.local_gpu_idx} | Iters: {args.iters}"
    )

    try:
        if rank == 0:
            _run_client(args)
        else:
            _run_server(args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] Benchmark aborted by user.")
        sys.exit(1)
