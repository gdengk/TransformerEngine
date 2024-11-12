# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""_A2ALinear API"""
from typing import Any, Callable, Dict, Optional, Tuple, Union, List

import torch

import transformer_engine_torch as tex
import os

from .base import (
    get_workspace,
    get_multi_stream_cublas_workspace,
    get_ub,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ._common import _noop_cat
from ..fp8 import get_fp8_te_dtype, FP8GlobalStateManager
from ..utils import (
    divide,
    cast_if_needed,
    assert_dim_for_fp8_exec,
    clear_tensor_data,
    init_method_constant,
    requires_grad,
)
from ..distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    allreduce,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
    _fsdp_scatter_tensors,
    _fsdp_gather_tensors,
    alltoall,
)
from ..cpp_extensions import (
    fp8_gemm,
    gemm,
    fp8_cast_transpose_fused,
    cast_to_fp8,
)
from ..constants import GemmParallelModes, dist_group_type
from ..jit import no_torch_dynamo
from ..graph import is_graph_capturing
from ..float8_tensor import Float8Tensor
from ..export import is_in_onnx_export_mode
from ..tensor import QuantizedTensor

import math

__all__ = ["_A2ALinear"]


def get_send_recv_ops(
    send_first,
    send_tensor,
    send_dst,
    recv_tensor,
    recv_src,
    process_group,
    batch_p2p_comm,
):
    """ Generate Pairs of P2P send/recv op.
        Swap the order of send and recv to avoid deadlock.
    """
    send_recv_ops = []
    if send_first:
        send_op = (
            torch.distributed.P2POp(
                torch.distributed.isend, send_tensor, send_dst, process_group
            )
            if batch_p2p_comm
            else torch.distributed.isend(send_tensor, send_dst, process_group)
        )
        recv_op = (
            torch.distributed.P2POp(
                torch.distributed.irecv, recv_tensor, recv_src, process_group
            )
            if batch_p2p_comm
            else torch.distributed.irecv(recv_tensor, recv_src, process_group)
        )
        send_recv_ops.append(send_op)
        send_recv_ops.append(recv_op)
    else:
        recv_op = (
            torch.distributed.P2POp(
                torch.distributed.irecv, recv_tensor, recv_src, process_group
            )
            if batch_p2p_comm
            else torch.distributed.irecv(recv_tensor, recv_src, process_group)
        )
        send_op = (
            torch.distributed.P2POp(
                torch.distributed.isend, send_tensor, send_dst, process_group
            )
            if batch_p2p_comm
            else torch.distributed.isend(send_tensor, send_dst, process_group)
        )
        send_recv_ops.append(recv_op)
        send_recv_ops.append(send_op)
    return send_recv_ops


def p2p_with_step(step, send_tensor, recv_tensor, process_group, batch_p2p_comm):
    """ 
    Send the send_tensor to the process with rank (local_rank + step) in the process_group.
    Receive the recv_tensor from the process with rank (local_rank - step) in the process_group.
    Assuming a circular topology where ranks wrap around at the boundaries.
    """
    assert step > 0, "the step has to be a positive num."

    pg_world_size = torch.distributed.get_world_size(process_group)
    local_rank = torch.distributed.get_rank(process_group)
    gcd = math.gcd(pg_world_size, step)
    send_first = (local_rank // gcd) % 2 == 0

    send_dst = torch.distributed.get_global_rank(
        process_group, (local_rank + step + pg_world_size) % pg_world_size
    )
    recv_src = torch.distributed.get_global_rank(
        process_group, (local_rank - step + pg_world_size) % pg_world_size
    )

    send_recv_ops = get_send_recv_ops(
        send_first,
        send_tensor,
        send_dst,
        recv_tensor,
        recv_src,
        process_group,
        batch_p2p_comm,
    )

    if batch_p2p_comm:
        send_recv_reqs = torch.distributed.batch_isend_irecv(send_recv_ops)
    else:
        send_recv_reqs = send_recv_ops
    return send_recv_reqs


def multi_steps_p2p(
    step_list,
    send_tensor_list,
    recv_tensor_list,
    sub_pg_size,
    process_group,
    batch_p2p_comm,
):
    """ 
    Performs multi-step point-to-point (p2p) communication within a given sub-process group.

    This function handles p2p communication operations for each step in the step_list 
    by sending and receiving tensors among processes in the process_group. Communication 
    occurs within a sub-process group of size sub_pg_size. 
    """

    local_rank = torch.distributed.get_rank(process_group)
    sub_pg_offset = local_rank // sub_pg_size
    sub_pg_rank = local_rank % sub_pg_size

    all_send_recv_ops = []

    for step, send_tensor, recv_tensor in zip(
        step_list, send_tensor_list, recv_tensor_list
    ):
        gcd = math.gcd(sub_pg_size, step)
        send_first = (sub_pg_rank // gcd) % 2 == 0
        send_dst = torch.distributed.get_global_rank(
            process_group,
            sub_pg_offset * sub_pg_size
            + (sub_pg_rank + step + sub_pg_size) % sub_pg_size,
        )
        recv_src = torch.distributed.get_global_rank(
            process_group,
            sub_pg_offset * sub_pg_size
            + (sub_pg_rank - step + sub_pg_size) % sub_pg_size,
        )

        all_send_recv_ops.extend(
            get_send_recv_ops(
                send_first,
                send_tensor,
                send_dst,
                recv_tensor,
                recv_src,
                process_group,
                batch_p2p_comm,
            )
        )

    if batch_p2p_comm:
        send_recv_reqs = torch.distributed.batch_isend_irecv(all_send_recv_ops)
    else:
        send_recv_reqs = all_send_recv_ops
    return send_recv_reqs


def aggregate_p2p(
    step,
    src_buf,
    dst_buf,
    aggregate_size,
    dst_buf_nonag_offset,
    process_group,
    batch_p2p_comm,
):
    """ 
    A helper function to perform multiple point-to-point (p2p) communications within a sub process group (i) to
    the destination process group (step + i) or from the source process group (i - step). 
    Assuming a circular topology where sub process group indices wrap around at the boundaries.

    Both src_buf and dst_buf are indexed using fine-grained, non-aggregated slices, 
    representing TP_size * EP_size in the original context.
    """
    pg_world_size = torch.distributed.get_world_size(process_group)
    local_rank = torch.distributed.get_rank(process_group)

    chunk_idx = local_rank // aggregate_size
    sub_chunk_rank = local_rank % aggregate_size
    chunk_size = pg_world_size // aggregate_size

    # send data to num of aggregate_size destinations
    send_recv_ops = []

    dst_chunk_idx = (chunk_idx + step) % chunk_size
    src_chunk_idx = (chunk_idx - step + chunk_size) % chunk_size

    gcd = math.gcd(chunk_size, step)
    send_first = (chunk_idx // gcd) % 2 == 0

    for i in range(aggregate_size):

        dst_rank = (
            dst_chunk_idx * aggregate_size + (i + sub_chunk_rank) % aggregate_size
        )
        src_rank = (
            src_chunk_idx * aggregate_size
            + (sub_chunk_rank - i + aggregate_size) % aggregate_size
        )

        send_tensor = src_buf[dst_rank]
        send_dst = torch.distributed.get_global_rank(process_group, dst_rank)
        recv_tensor = dst_buf[src_rank + dst_buf_nonag_offset]
        recv_src = torch.distributed.get_global_rank(process_group, src_rank)

        send_recv_ops.extend(
            get_send_recv_ops(
                send_first,
                send_tensor,
                send_dst,
                recv_tensor,
                recv_src,
                process_group,
                batch_p2p_comm,
            )
        )

    if batch_p2p_comm:
        send_recv_reqs = torch.distributed.batch_isend_irecv(send_recv_ops)
    else:
        send_recv_reqs = send_recv_ops
    return send_recv_reqs


def ring_exchange_overlap_ag(
    weight,
    input,
    activation_dtype,
    out,
    ub_algo,
    ub_obj,
    extra_output_tensor,
    ep_group,
    ep_size,
    tp_group,
    tp_size,
    layout="TN",
    grad=False,
    external_streams=None,
):
    """ 
    Multi-stream A2A AG and GEMM overlap using the ring-exchange method.
    """
    # use batched P2P as default to reduce num of SMs taken
    batch_p2p_comm = True

    ep_local_rank = torch.distributed.get_rank(ep_group) if ep_size != 1 else 0
    tp_local_rank = torch.distributed.get_rank(tp_group) if tp_size != 1 else 0

    ep_reqs = []
    tp_reqs = []

    if external_streams is not None:
        streams = external_streams
    else:
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    chunked_input = torch.chunk(input, chunks=ep_size, dim=0)
    # tmp buf reserved after AG
    comm_full_buf = torch.empty(
        (input.shape[0] * tp_size),
        input.shape[1],
        dtype=input.dtype,
        device=input.device,
    )
    comm_buf = torch.chunk(comm_full_buf, chunks=ep_size * tp_size, dim=0)
    assert (
        out is not None
    ), "must reserve output space for ring-exchange based A2A-AG overlap"
    chunked_out = torch.chunk(out, chunks=ep_size * tp_size, dim=0)
    assert ub_obj is None, "ring_exchange_overlap_ag does not support UB for now"
    assert (
        extra_output_tensor is None
    ), "ring_exchange_overlap_ag does not support extra_output_tensor from UB"

    # copy the local data slice to comm_buf
    # (TODO:) get rid of this local D2D
    comm_buf_offset = tp_local_rank * ep_size
    comm_buf[ep_local_rank + comm_buf_offset].copy_(chunked_input[ep_local_rank])

    cublasworkspace = get_multi_stream_cublas_workspace()

    for s in streams:
        s.wait_stream(torch.cuda.current_stream())

    for i in range(ep_size):
        src_ep_i = (ep_local_rank + i + 1) % ep_size
        dst_ep_i = (ep_local_rank - 1 - i + ep_size) % ep_size
        cur_ep_i = (ep_local_rank - i + ep_size) % ep_size
        for j in range(tp_size):
            streamid = (i * tp_size + j) % 2
            with torch.cuda.stream(streams[streamid]):
                if j == 0:
                    for req in ep_reqs:
                        req.wait()

                    if i < ep_size - 1:
                        dst_comm_buf_i = dst_ep_i + tp_local_rank * ep_size
                        ep_reqs = p2p_with_step(
                            i + 1,
                            chunked_input[src_ep_i],
                            comm_buf[dst_comm_buf_i],
                            ep_group,
                            batch_p2p_comm,
                        )

                for req in tp_reqs:
                    req.wait()

                cur_tp_i = (tp_local_rank - j + tp_size) % tp_size
                dst_tp_i = (tp_local_rank - j - 1 + tp_size) % tp_size
                src_comm_buf_i = cur_tp_i * ep_size + cur_ep_i
                if j < tp_size - 1:
                    dst_comm_buf_i = dst_tp_i * ep_size + cur_ep_i
                    tp_reqs = p2p_with_step(
                        1,
                        comm_buf[src_comm_buf_i],
                        comm_buf[dst_comm_buf_i],
                        tp_group,
                        batch_p2p_comm,
                    )
                else:
                    tp_reqs = []

                _ = gemm(
                    weight,
                    comm_buf[src_comm_buf_i],
                    activation_dtype,
                    cublasworkspace[streamid],
                    out=chunked_out[src_comm_buf_i],
                    ub_algo=ub_algo,
                    ub=ub_obj,
                    extra_output_tensor=extra_output_tensor,
                    layout=layout,
                    grad=grad,
                )

    for s in streams:
        torch.cuda.current_stream().wait_stream(s)

    dim_size = comm_buf[0].size(0)
    a2a_out = comm_full_buf[
        comm_buf_offset * dim_size : (comm_buf_offset + ep_size) * dim_size
    ]
    ag_out = comm_full_buf

    return out, a2a_out, ag_out


def ring_exchange_overlap_ag_aggregate(
    weight,
    input,
    activation_dtype,
    out,
    ub_algo,
    ub_obj,
    extra_output_tensor,
    ep_aggregate,
    tp_aggregate,
    ep_group,
    ep_size,
    tp_group,
    tp_size,
    layout="TN",
    grad=False,
    external_streams=None,
):
    """ 
    Multi-stream A2A AG and GEMM overlap using the ring-exchange method.
    Aggregate EP steps to reduce the total num of chunks by this method.
    """
    # use batched P2P as default to reduce num of SMs taken
    batch_p2p_comm = True

    ep_local_rank = torch.distributed.get_rank(ep_group) if ep_size != 1 else 0
    tp_local_rank = torch.distributed.get_rank(tp_group) if tp_size != 1 else 0

    if ep_aggregate is None or ep_size == 1:
        ep_aggregate = 1

    assert ep_size % ep_aggregate == 0, "ep_size should be divisible by ep_aggregate."

    ep_chunksize = ep_size // ep_aggregate
    # tp aggregate will potentially affect the order of data.
    # (TODO) add support for tp_aggregate.
    tp_chunksize = tp_size
    assert (
        tp_aggregate is None or tp_aggregate == 1
    ), "Currently this function does not support tp aggreate."

    ep_reqs = []
    tp_reqs = []

    # two streams for now
    if external_streams is not None:
        streams = external_streams
    else:
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    chunked_input = torch.chunk(input, chunks=ep_size, dim=0)
    # tmp buf reserved after AG
    comm_full_buf = torch.empty(
        (input.shape[0] * tp_size),
        input.shape[1],
        dtype=input.dtype,
        device=input.device,
    )
    # chunk the comm_full_buf in the aggregated view
    comm_buf_ag_view = torch.chunk(
        comm_full_buf, chunks=ep_chunksize * tp_chunksize, dim=0
    )
    # chunk the comm_full_buf in a more fine-grained, non-aggregated view
    comm_buf_nonag_view = torch.chunk(comm_full_buf, chunks=ep_size * tp_size, dim=0)

    assert (
        out is not None
    ), "must reserve output space for ring-exchange based A2A-AG overlap"
    chunked_out = torch.chunk(out, chunks=ep_chunksize * tp_chunksize, dim=0)
    assert (
        ub_obj is None
    ), "ring_exchange_overlap_ag_aggregate does not support UB for now"
    assert (
        extra_output_tensor is None
    ), "ring_exchange_overlap_ag_aggregate does not support extra_output_tensor from UB"

    comm_buf_nonag_offset = tp_local_rank * ep_size
    comm_buf_ag_offset = tp_local_rank * ep_chunksize
    ep_chunk_idx = ep_local_rank // ep_aggregate

    cublasworkspace = get_multi_stream_cublas_workspace()
    for s in streams:
        s.wait_stream(torch.cuda.current_stream())

    # preprocess the very first aggregated EP input data
    if ep_aggregate > 1:
        sub_pg_offset = ep_chunk_idx * ep_aggregate
        sub_pg_rank = ep_local_rank % ep_aggregate
        preprocess_step_list = list(range(1, ep_aggregate))
        preprocess_send_tensor_list = []
        preprocess_recv_tensor_list = []
        for i in preprocess_step_list:
            sub_send_src_rank = sub_pg_offset + (sub_pg_rank + i) % ep_aggregate
            sub_recv_dst_rank = (
                sub_pg_offset + (sub_pg_rank - i + ep_aggregate) % ep_aggregate
            )
            preprocess_send_tensor_list.append(chunked_input[sub_send_src_rank])
            preprocess_recv_tensor_list.append(
                comm_buf_nonag_view[sub_recv_dst_rank + comm_buf_nonag_offset]
            )

        ep_reqs = multi_steps_p2p(
            preprocess_step_list,
            preprocess_send_tensor_list,
            preprocess_recv_tensor_list,
            ep_aggregate,
            ep_group,
            batch_p2p_comm,
        )

    # copy the local data slice to comm_buf
    comm_buf_nonag_view[ep_local_rank + comm_buf_nonag_offset].copy_(
        chunked_input[ep_local_rank]
    )

    for i in range(ep_chunksize):
        cur_ep_i = (ep_chunk_idx - i + ep_chunksize) % ep_chunksize
        for j in range(tp_chunksize):
            streamid = (i * tp_chunksize + j) % 2
            with torch.cuda.stream(streams[streamid]):
                if j == 0:
                    for req in ep_reqs:
                        req.wait()

                    if i < ep_chunksize - 1:
                        ep_reqs = aggregate_p2p(
                            i + 1,
                            chunked_input,
                            comm_buf_nonag_view,
                            ep_aggregate,
                            comm_buf_nonag_offset,
                            ep_group,
                            batch_p2p_comm,
                        )

                # Start inner TP ring exchange
                for req in tp_reqs:
                    req.wait()

                cur_tp_i = (tp_local_rank - j + tp_size) % tp_size
                dst_tp_i = (tp_local_rank - j - 1 + tp_size) % tp_size
                src_comm_buf_i = cur_tp_i * ep_chunksize + cur_ep_i
                if j < tp_chunksize - 1:
                    dst_comm_buf_i = dst_tp_i * ep_chunksize + cur_ep_i
                    tp_reqs = p2p_with_step(
                        1,
                        comm_buf_ag_view[src_comm_buf_i],
                        comm_buf_ag_view[dst_comm_buf_i],
                        tp_group,
                        batch_p2p_comm,
                    )
                else:
                    tp_reqs = []

                _ = gemm(
                    weight,
                    comm_buf_ag_view[src_comm_buf_i],
                    activation_dtype,
                    cublasworkspace[streamid],
                    out=chunked_out[src_comm_buf_i],
                    ub_algo=ub_algo,
                    ub=ub_obj,
                    extra_output_tensor=extra_output_tensor,
                    layout=layout,
                    grad=grad,
                )

    for s in streams:
        torch.cuda.current_stream().wait_stream(s)

    dim_size = comm_buf_nonag_view[0].size(0)
    a2a_out = comm_full_buf[
        comm_buf_nonag_offset * dim_size : (comm_buf_nonag_offset + ep_size) * dim_size
    ]

    return out, a2a_out, comm_full_buf


def ring_exchange_overlap_rs(
    weight,
    input,
    activation_dtype,
    out,
    ub_algo,
    ub_obj,
    extra_output_tensor,
    ep_aggregate2,
    tp_aggregate2,
    ep_group,
    ep_size,
    tp_group,
    tp_size,
    external_streams=None,
):
    """ 
    GEMM, RS and A2A overlap using ring-exchange method.
    Use UB for TP communication overlap. Not support any TP/EP aggregate right now.
    """
    batch_p2p_comm = True
    ep_local_rank = torch.distributed.get_rank(ep_group) if ep_size != 1 else 0

    assert (
        not ep_aggregate2 and not tp_aggregate2
    ), "not support any TP/EP aggregate for now."
    ep_aggregate2 = False if ep_size == 1 else ep_aggregate2
    ep_chunksize = ep_size // 2 if ep_aggregate2 else ep_size

    ep_reqs = []

    chunked_input = torch.chunk(input, chunks=ep_size * tp_size, dim=0)
    # Not using UB when TP_size = 1
    if tp_size != 1:
        chunked_out = torch.chunk(extra_output_tensor, chunks=ep_size, dim=0)
    else:
        chunked_out = torch.chunk(out, chunks=ep_size, dim=0)

    rs_out_fullbuf = torch.empty(
        (chunked_out[0].size(0) * ep_size, chunked_out[0].size(1)),
        dtype=chunked_out[0].dtype,
        device=chunked_out[0].device,
    )
    rs_out = torch.chunk(rs_out_fullbuf, chunks=ep_size, dim=0)

    # continous input for user buf
    gemm_in_array = [
        torch.cat(chunked_input[i::ep_size], dim=0) for i in range(ep_size)
    ]

    for i in range(ep_chunksize):
        cur_ep_i = (ep_local_rank + i + 1) % ep_size
        dst_ep_i = (ep_local_rank - i - 1 + ep_size) % ep_size

        if i == ep_chunksize - 1:
            gemm_rs_out = chunked_out[dst_ep_i]
        else:
            gemm_rs_out = rs_out[cur_ep_i]
        if tp_size == 1:
            gemm_out = gemm_rs_out
            gemm_extra_output_tensor = None
        else:
            gemm_out = ub_obj.get_ubuf_output(1)
            gemm_extra_output_tensor = gemm_rs_out

        _ = gemm(
            weight,
            gemm_in_array[cur_ep_i],
            activation_dtype,
            get_workspace(),
            out=gemm_out,
            ub_algo=ub_algo,
            ub=ub_obj,
            extra_output_tensor=gemm_extra_output_tensor,
        )

        for req in ep_reqs:
            req.wait()
        if i < ep_chunksize - 1:
            ep_reqs = p2p_with_step(
                i + 1, rs_out[cur_ep_i], chunked_out[dst_ep_i], ep_group, batch_p2p_comm
            )

    return extra_output_tensor


def ring_exchange_overlap_rs_tp1(
    weight,
    input,
    activation_dtype,
    out,
    ub_algo,
    ub_obj,
    extra_output_tensor,
    ep_aggregate2,
    tp_aggregate2,
    ep_group,
    ep_size,
    tp_group,
    tp_size,
    external_streams=None,
):
    """ 
    Multi-stream GEMM and A2A overlap using ring-exchange method when TP=1 only.
    Not using UB makes multi-stream possible to potentially hide the gemm quantization effect.
    """
    batch_p2p_comm = True
    assert tp_size == 1, "only support TP1 in this case"
    ep_local_rank = torch.distributed.get_rank(ep_group) if ep_size != 1 else 0

    if external_streams is not None:
        streams = external_streams
    else:
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    assert (
        not ep_aggregate2 and not tp_aggregate2
    ), "not support any TP/EP aggregate for now."
    ep_aggregate2 = False if ep_size == 1 else ep_aggregate2
    ep_chunksize = ep_size // 2 if ep_aggregate2 else ep_size

    ep_reqs = []

    chunked_input = torch.chunk(input, chunks=ep_size * tp_size, dim=0)
    chunked_out = torch.chunk(out, chunks=ep_size, dim=0)
    rs_out = [torch.empty_like(chunked_out[0]) for _ in range(ep_size - 1)]

    cublasworkspace = get_multi_stream_cublas_workspace()

    for s in streams:
        s.wait_stream(torch.cuda.current_stream())

    for i in range(ep_chunksize):
        cur_ep_i = (ep_local_rank + i + 1) % ep_size
        dst_ep_i = (ep_local_rank - i - 1 + ep_size) % ep_size

        with torch.cuda.stream(streams[i % 2]):

            if i == ep_chunksize - 1:
                gemm_rs_out = chunked_out[dst_ep_i]
            else:
                gemm_rs_out = rs_out[i]
            gemm_in = chunked_input[cur_ep_i]

            _ = gemm(
                weight,
                gemm_in,
                activation_dtype,
                cublasworkspace[i % 2],
                out=gemm_rs_out,
                ub_algo=ub_algo,
                ub=ub_obj,
                extra_output_tensor=None,
            )

            if i < ep_chunksize - 1:
                reqs = p2p_with_step(
                    i + 1, rs_out[i], chunked_out[dst_ep_i], ep_group, batch_p2p_comm
                )
                ep_reqs.extend(reqs)
    for s in streams:
        torch.cuda.current_stream().wait_stream(s)

    for req in ep_reqs:
        req.wait()

    return out


def rs_a2a_bulk_overlap_gemm(
    weight,
    input,
    activation_dtype,
    ep_group,
    ep_size,
    tp_group,
    tp_size,
    layout="TN",
    grad=False,
    rs_in=None,
    accumulate_wgrad_into_param_main_grad=False,
    out=None,
    external_streams=None,
):
    """ 
    Pipeline overlap between RS and A2A, then communication is bulk overlapped with GEMM.
    The gemm input is in the order of [TP_size, EP_size, ...]
    In this method, RS is splitted on the hidden dim.
    """
    if external_streams is not None:
        streams = external_streams
    else:
        streams = [torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()]

    # (TODO) expose this chunk_size later
    if tp_size == 1 or ep_size == 1:
        chunk_size = 1
        chunked_rs_in = [rs_in]
    else:
        chunk_size = 2
        chunked_rs_in = list(torch.chunk(rs_in, chunks=chunk_size, dim=1))
        # original_sm = os.environ['NVTE_EXT_MARGIN_SM'] if 'NVTE_EXT_MARGIN_SM' in os.environ else '0'
        # os.environ['NVTE_EXT_MARGIN_SM'] = '24'
    a2a_output = [
        torch.empty(
            rs_in.size(0) // tp_size,
            rs_in.size(1) // chunk_size,
            dtype=rs_in.dtype,
            device=rs_in.device,
        )
        for _ in range(chunk_size)
    ]
    a2a_in = [
        torch.empty(
            rs_in.size(0) // tp_size,
            rs_in.size(1) // chunk_size,
            dtype=rs_in.dtype,
            device=rs_in.device,
        )
        for _ in range(chunk_size)
    ]

    rs_handles = []
    a2a_handles = []

    for s in streams:
        s.wait_stream(torch.cuda.current_stream())

    for i in range(chunk_size):
        with torch.cuda.stream(streams[0]):
            if tp_size != 1:
                rs_handle = torch.distributed._reduce_scatter_base(
                    a2a_in[i],
                    chunked_rs_in[i].contiguous(),
                    group=tp_group,
                    async_op=True,
                )
                rs_handles.append(rs_handle)
                if i + 1 < chunk_size:
                    chunked_rs_in[i + 1] = chunked_rs_in[i + 1].contiguous()

            else:
                a2a_in = chunked_rs_in
                rs_handles = []

        with torch.cuda.stream(streams[1]):
            if ep_size != 1:
                if len(rs_handles) != 0:
                    rs_handles[i].wait()
                a2a_handle = torch.distributed.all_to_all_single(
                    a2a_output[i], a2a_in[i], None, None, ep_group, async_op=True,
                )
                a2a_handles.append(a2a_handle)
            else:
                a2a_output = a2a_in
                a2a_handles = rs_handles

    with torch.cuda.stream(streams[2]):
        wgrad, grad_bias, _ = gemm(
            weight,
            input,
            activation_dtype,
            get_workspace(),
            layout=layout,
            grad=grad,
            use_bias=False,
            accumulate=accumulate_wgrad_into_param_main_grad,
            out=out,
        )

    for a2a_handle in a2a_handles:
        a2a_handle.wait()

    for s in streams:
        torch.cuda.current_stream().wait_stream(s)

    if tp_size == 1 or ep_size == 1:
        rs_out = a2a_output[0]
    else:
        rs_out = torch.cat(a2a_output, dim=1)

    # if tp_size != 1 and ep_size != 1:
    #     os.environ['NVTE_EXT_MARGIN_SM'] = original_sm
    return wgrad, grad_bias, rs_out


########## pipeline split algorithms ##########


def pipeline_split_ag(
    weight,
    input,
    activation_dtype,
    out,
    ub_algo,
    ub_obj,
    extra_output_tensor,
    num_pipeline_stage,
    ep_group,
    ep_size,
    tp_group,
    tp_size,
    layout="TN",
    grad=False,
    external_streams=None,
):
    """ 
    Mutil-stream pipeline split overlap for A2A, AG and GEMM.
    This algorithm needs input data to be permuted in the order of
    [num_split_stage, TP_size, ...]
    """
    chunked_input = torch.chunk(input, chunks=num_pipeline_stage, dim=0)
    if ep_size != 1:
        a2a_full_buf = torch.empty_like(input)
        a2a_out = torch.chunk(a2a_full_buf, chunks=num_pipeline_stage, dim=0)
    else:
        a2a_full_buf = input
        a2a_out = chunked_input

    if tp_size != 1:
        ag_out_full_buf = torch.empty(
            (input.size(0) * tp_size, input.size(1)),
            dtype=input.dtype,
            device=input.device,
        )
        ag_out = torch.chunk(ag_out_full_buf, chunks=num_pipeline_stage, dim=0)
    else:
        ag_out_full_buf = a2a_full_buf
        ag_out = a2a_out

    chunked_output = torch.chunk(out, chunks=num_pipeline_stage, dim=0)

    cublasworkspace = get_multi_stream_cublas_workspace()

    if external_streams is not None:
        streams = external_streams
    else:
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    a2a_handles = [None, None]
    ag_handles = [None, None]

    for s in streams:
        s.wait_stream(torch.cuda.current_stream())

    extra_stage = 2 if tp_size != 1 else 1
    for i in range(num_pipeline_stage + extra_stage):
        with torch.cuda.stream(streams[i % 2]):

            if a2a_handles[(i + 1) % 2] is not None:
                a2a_handles[(i + 1) % 2].wait()

            if i < num_pipeline_stage:
                if ep_size != 1:
                    a2a_handles[i % 2] = torch.distributed.all_to_all_single(
                        a2a_out[i],
                        chunked_input[i],
                        None,
                        None,
                        ep_group,
                        async_op=True,
                    )

            if ag_handles[(i + 1) % 2] is not None:
                ag_handles[(i + 1) % 2].wait()
            if i > 0 and i < num_pipeline_stage + 1:
                if tp_size != 1:
                    ag_handles[i % 2] = torch.distributed._all_gather_base(
                        ag_out[i - 1], a2a_out[i - 1], group=tp_group, async_op=True
                    )

            if i > extra_stage - 1:
                _ = gemm(
                    weight,
                    ag_out[i - extra_stage],
                    activation_dtype,
                    cublasworkspace[i % 2],
                    out=chunked_output[i - extra_stage],
                    ub_algo=ub_algo,
                    ub=ub_obj,
                    extra_output_tensor=extra_output_tensor,
                    layout=layout,
                    grad=grad,
                )

    for s in streams:
        torch.cuda.current_stream().wait_stream(s)

    return out, a2a_full_buf, ag_out_full_buf


def pipeline_split_rs(
    weight,
    input,
    activation_dtype,
    out,
    ub_algo,
    ub_obj,
    extra_output_tensor,
    num_pipeline_stage,
    ep_group,
    ep_size,
    tp_group,
    tp_size,
    external_streams=None,
):
    """ 
    Mutil-stream pipeline split overlap for GEMM, RS and A2A.
    This algorithm needs input data to be permuted in the order of
    [num_split_stage, TP_size, EP_size, ...]
    """
    chunked_input = torch.chunk(input, chunks=num_pipeline_stage, dim=0)
    a2a_out = list(torch.chunk(out, chunks=num_pipeline_stage, dim=0))
    if ep_size == 1:
        rs_out_full_buf = out
        rs_output = a2a_out
    else:
        rs_out_full_buf = torch.empty_like(out)
        rs_output = list(torch.chunk(rs_out_full_buf, chunks=num_pipeline_stage, dim=0))

    if tp_size == 1:
        gemm_out_full_buf = rs_out_full_buf
        rs_input = rs_output
    else:
        gemm_out_full_buf = torch.empty(
            (input.size(0), out.size(1)), dtype=input.dtype, device=input.device
        )
        rs_input = torch.chunk(gemm_out_full_buf, chunks=num_pipeline_stage, dim=0)

    cublasworkspace = get_multi_stream_cublas_workspace()

    if external_streams is not None:
        streams = external_streams
    else:
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    a2a_handles = [None, None]
    rs_handles = [None, None]
    assert ub_algo is None and ub_obj is None, "does not support UB in this algorithm"

    for s in streams:
        s.wait_stream(torch.cuda.current_stream())

    extra_stage = 1 if tp_size != 1 else 0
    for i in range(num_pipeline_stage + extra_stage):
        with torch.cuda.stream(streams[i % 2]):
            if i < num_pipeline_stage:
                _ = gemm(
                    weight,
                    chunked_input[i],
                    activation_dtype,
                    cublasworkspace[i % 2],
                    out=rs_input[i],
                    ub_algo=ub_algo,
                    ub=ub_obj,
                    extra_output_tensor=extra_output_tensor,
                )

            if rs_handles[(i + 1) % 2] is not None:
                rs_handles[(i + 1) % 2].wait()
            if i < num_pipeline_stage:
                if tp_size != 1:
                    rs_handles[i % 2] = torch.distributed._reduce_scatter_base(
                        rs_output[i], rs_input[i], group=tp_group, async_op=True
                    )

            if a2a_handles[(i + 1) % 2] is not None:
                a2a_handles[(i + 1) % 2].wait()
            if i >= extra_stage and i < num_pipeline_stage + extra_stage:
                if ep_size != 1:
                    a2a_handles[i % 2] = torch.distributed.all_to_all_single(
                        a2a_out[i - extra_stage],
                        rs_output[i - extra_stage],
                        None,
                        None,
                        ep_group,
                        async_op=True,
                    )

    for handle in a2a_handles + rs_handles:
        if handle is not None:
            handle.wait()

    for s in streams:
        torch.cuda.current_stream().wait_stream(s)

    return out


def pipeline_split_bulk_ag(
    weight,
    input,
    activation_dtype,
    out,
    ub_algo,
    ub_obj,
    extra_output_tensor,
    num_pipeline_stage,
    ep_group,
    ep_size,
    tp_group,
    tp_size,
    layout="TN",
    grad=False,
    ag_in=None,
    external_streams=None,
):
    """
    Bulk overlap for AG and GEMM.
    The AG input has already been permuted into the order of [num_split_stage, EP_size, ...],
    therefore AG needs to be pipelined into multiple stages.
    """
    ag_out_full_buf = torch.empty(
        (ag_in.size(0) * tp_size, ag_in.size(1)), dtype=ag_in.dtype, device=ag_in.device
    )

    chunked_ag_in = torch.chunk(ag_in, chunks=num_pipeline_stage, dim=0)
    chunked_ag_out = torch.chunk(ag_out_full_buf, chunks=num_pipeline_stage, dim=0)

    if external_streams is not None:
        streams = external_streams
    else:
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    for s in streams:
        s.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(streams[1]):
        # use SM carveout to make better overlap of communication and computation
        # num_SM_prev = (
        #     os.environ["NVTE_EXT_MARGIN_SM"]
        #     if "NVTE_EXT_MARGIN_SM" in os.environ
        #     else "0"
        # )
        # os.environ["NVTE_EXT_MARGIN_SM"] = "16"
        _, _, _ = gemm(
            weight,
            input,
            activation_dtype,
            get_workspace(),
            layout=layout,
            grad=grad,
            out=out,
        )
        # os.environ["NVTE_EXT_MARGIN_SM"] = num_SM_prev

    ag_handle = None
    with torch.cuda.stream(streams[0]):
        if tp_size != 1:
            for i in range(num_pipeline_stage):
                if ag_handle is not None:
                    ag_handle.wait()
                ag_handle = torch.distributed._all_gather_base(
                    chunked_ag_out[i], chunked_ag_in[i], group=tp_group, async_op=True
                )
        else:
            ag_out_full_buf = ag_in

    if ag_handle is not None:
        ag_handle.wait()

    for s in streams:
        torch.cuda.current_stream().wait_stream(s)

    return ag_out_full_buf


def pipeline_split_bulk_rs_a2a(
    weight,
    input,
    activation_dtype,
    num_pipeline_stage,
    ep_group,
    ep_size,
    tp_group,
    tp_size,
    layout="TN",
    grad=False,
    rs_in=None,
    accumulate_wgrad_into_param_main_grad=False,
    out=None,
    external_streams=None,
):
    """ 
    Pipeline overlap between RS and A2A, then communication is bulk overlapped with GEMM.
    The gemm input is in the order of [num_split_stage, TP_size, EP_size, ...]
    """
    chunked_rs_in = torch.chunk(rs_in, chunks=num_pipeline_stage, dim=0)

    if tp_size == 1:
        rs_out_full_buf = rs_in
        chunked_rs_out = chunked_rs_in
    else:
        rs_out_full_buf = torch.empty(
            (rs_in.size(0) // tp_size, rs_in.size(1)),
            dtype=rs_in.dtype,
            device=rs_in.device,
        )
        chunked_rs_out = torch.chunk(rs_out_full_buf, chunks=num_pipeline_stage, dim=0)

    if ep_size == 1:
        a2a_output = rs_out_full_buf
        chunked_a2a_output = chunked_rs_out
    else:
        a2a_output = torch.empty(
            (rs_in.size(0) // tp_size, rs_in.size(1)),
            dtype=rs_in.dtype,
            device=rs_in.device,
        )
        chunked_a2a_output = list(
            torch.chunk(a2a_output, chunks=num_pipeline_stage, dim=0)
        )

    if external_streams is not None:
        streams = external_streams
    else:
        streams = [torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()]
    rs_handles = []
    a2a_handles = []

    for s in streams:
        s.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(streams[2]):
        wgrad, grad_bias, _ = gemm(
            weight,
            input,
            activation_dtype,
            get_workspace(),
            layout=layout,
            grad=grad,
            use_bias=False,
            accumulate=accumulate_wgrad_into_param_main_grad,
            out=out,
        )

    for i in range(num_pipeline_stage):
        with torch.cuda.stream(streams[0]):
            if tp_size != 1:
                rs_handle = torch.distributed._reduce_scatter_base(
                    chunked_rs_out[i], chunked_rs_in[i], group=tp_group, async_op=True,
                )
                rs_handles.append(rs_handle)

        with torch.cuda.stream(streams[1]):
            if ep_size != 1:
                if len(rs_handles) != 0:
                    rs_handles[i].wait()
                a2a_handle = torch.distributed.all_to_all_single(
                    chunked_a2a_output[i],
                    chunked_rs_out[i],
                    None,
                    None,
                    ep_group,
                    async_op=True,
                )
                a2a_handles.append(a2a_handle)

    for a2a_handle in a2a_handles:
        a2a_handle.wait()

    for s in streams:
        torch.cuda.current_stream().wait_stream(s)

    return wgrad, grad_bias, a2a_output


class _A2ALinear(torch.autograd.Function):
    """Linear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        weight: Union[Float8Tensor, torch.Tensor],
        weight_fp8: Optional[Float8Tensor],
        inp: torch.Tensor,
        bias: torch.Tensor,
        use_bias: bool,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        fp8_meta: Dict[str, Any],
        fuse_wgrad_accumulation: bool,
        cpu_offloading: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        parallel_mode: Union[str, None],
        is_grad_enabled: bool,
        ub_overlap_rs: bool,
        ub_overlap_ag: bool,
        ub_name: str,
        fp8_output: bool,
        fsdp_group: Union[dist_group_type, None],
        ep_group: Union[dist_group_type, None],
        ep_size: int,
        moe_use_a2alinear: bool,
        moe_ring_exchange: bool,
        moe_pipeline_split: bool,
        moe_num_pipeline_stage: int,
        ep_streams: Union[List[torch.cuda.Stream], None],
    ) -> torch.Tensor:
        is_input_fp8 = isinstance(inp, Float8Tensor)

        # Make sure input dimensions are compatible
        in_features = weight.shape[-1]
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view(-1, in_features)
        if fp8:
            assert_dim_for_fp8_exec(inputmat)
            assert_dim_for_fp8_exec(weight)

        tp_world_size = get_distributed_world_size(tp_group)
        ub_overlap_rs = False if tp_world_size == 1 else ub_overlap_rs
        ub_overlap_ag = False if tp_world_size == 1 else ub_overlap_ag

        # MoE alltoall overlap control options
        disable_a2a_ag_overlap = False
        disable_rs_ag_overlap = False
        # here moe_use_a2alinear means move a2a and ag into TE side, does not necessarily means they are going to be overlap
        assert (not moe_ring_exchange) or (
            not moe_pipeline_split
        ), "moe_ring_exchange and moe_pipeline_split could not both be true"
        a2a_ag_overlap = (
            (moe_ring_exchange or moe_pipeline_split)
            and (parallel_mode == "column")
            and (not disable_a2a_ag_overlap)
        )
        # (TODO) disable ring-exchange rs_a2a_overlap for now since mult-instance UB has accuracy problem
        rs_a2a_overlap = (
            moe_pipeline_split
            and (parallel_mode == "row")
            and (not disable_rs_ag_overlap)
        )

        num_pipeline_stage = moe_num_pipeline_stage
        #(TODO) expose ep_aggregate_size later
        ep_aggregate_size = 1

        # disable UB when appropriate
        if a2a_ag_overlap:
            ub_overlap_ag = False
        if moe_pipeline_split:
            ub_overlap_rs = False
            ub_overlap_ag = False

        # get the UB object from its naem
        if (ub_overlap_ag and parallel_mode == "column") or (
            ub_overlap_rs and parallel_mode == "row"
        ):
            ub_obj = get_ub(ub_name + "_fprop")
        else:
            ub_obj = None

        if (
            moe_use_a2alinear
            and ep_size != 1
            and not a2a_ag_overlap
            and parallel_mode == "column"
        ):
            if ub_overlap_ag:
                a2a_out = ub_obj.get_ubuf_output(0)
            else:
                a2a_out = None

            inputmat, _ = alltoall(
                a2a_out,
                inputmat,
                None,  # input_splits
                None,  # output_splits
                ep_group,
                False,  # synchronous operation
            )

        # Cast input to expected dtype
        # (TODO: guard something here to avoid conversion on ub version inputmat )
        inputmat = cast_if_needed(inputmat, activation_dtype)
        inputmat_t = None
        inputmat_no_fp8 = inputmat
        inputmat_scale_inv = None

        if fp8:
            fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
            if isinstance(inputmat, Float8Tensor):
                inputmat_scale_inv = inputmat._scale_inv
            else:
                inputmat_scale_inv = torch.empty(
                    [1], dtype=torch.float32, device=inputmat.device
                )
                if (
                    not fp8_meta["recipe"].override_linear_precision.wgrad
                    and is_grad_enabled
                    and weight.requires_grad
                    and not sequence_parallel
                ):
                    # FP8 input for forward, FP8 input transpose for backward wgrad
                    inputmat, inputmat_t = fp8_cast_transpose_fused(
                        inputmat,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_INPUT,
                        fp8_dtype_forward,
                        scale_inv=inputmat_scale_inv,
                    )
                else:
                    # FP8 input for forward
                    inputmat = cast_to_fp8(
                        inputmat,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_INPUT,
                        fp8_dtype_forward,
                        scale_inv=inputmat_scale_inv,
                    )

            # Hack for ONNX export
            # Note: ONNX models are represented as a graph of tensor
            # operations, so the in-place scale-inv update doesn't fit
            # very well. We work around this by making it look like
            # the scale-inv tensor is initialized with a copy.
            # Note: ONNX export expects FP8 scales can be represented
            # with constant ops. However, copying into a buffer
            # involves an expand op for array broadcasting. We work
            # around this by filling the buffer instead.
            if is_in_onnx_export_mode():
                inputmat_scale_inv.fill_(inputmat_scale_inv.item())

        # Column Parallel Linear
        if parallel_mode == "column" and sequence_parallel and not a2a_ag_overlap:
            if ub_overlap_ag and not fp8:
                # (TODO:) support fp8 later.
                dim_size = list(inputmat.size())
                dim_size[0] = dim_size[0] * tp_world_size
                inputmat_total = ub_obj.get_ubuf_output(1)
            else:
                inputmat_total, _ = gather_along_first_dim(inputmat, tp_group)
        else:
            # if a2a_ag_overlap, A2A and AG will be performed altogether later
            inputmat_total = inputmat
        if fp8:
            bias_dtype = (
                torch.bfloat16
                if activation_dtype == torch.float32
                else activation_dtype
            )
            bias = cast_if_needed(bias, bias_dtype) if use_bias else bias

            # Use FP8 weights
            if weight_fp8 is None:
                weight_fp8 = weight

            assert isinstance(weight_fp8, Float8Tensor)

            if fp8_output:
                proj_out_index, meta_tensor, proj_out_tetype, proj_out_pttype = (
                    tex.FP8FwdTensors.GEMM1_OUTPUT,
                    fp8_meta["scaling_fwd"],
                    fp8_dtype_forward,
                    torch.uint8,
                )
            else:
                proj_out_index, meta_tensor, proj_out_tetype, proj_out_pttype = (
                    None,
                    None,
                    None,
                    activation_dtype,
                )

            if ub_overlap_rs:
                out = ub_obj.get_ubuf_output(1)
                dim_size = list(inputmat_total.size())
                dim_size[0] = dim_size[0] // tp_world_size
                dim_size[1] = weight_fp8.size(0)
                rs_out = torch.empty(
                    dim_size, dtype=activation_dtype, device=inputmat_total.device
                )
                if ub_obj.is_p2p_overlap():
                    if ub_obj.is_atomic_gemm():
                        ub_algo = tex.UbufOverlapAlgo.ATOMIC_GEMM_RS_P2P
                    else:
                        ub_algo = tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS_P2P
                else:
                    if ub_obj.is_atomic_gemm():
                        ub_algo = tex.UbufOverlapAlgo.ATOMIC_GEMM_RS
                    else:
                        ub_algo = tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS
                if ub_obj.is_fp8_ubuf():
                    proj_out_index = tex.FP8FwdTensors.GEMM1_OUTPUT
                    meta_tensor = fp8_meta["scaling_fwd"]
                    proj_out_tetype = fp8_dtype_forward
                    proj_out_pttype = torch.uint8
                    ub_obj.set_ubuf_scale_inv(meta_tensor.scale_inv[proj_out_index])
            else:
                dim_size = list(inputmat_total.size())
                dim_size[1] = weight_fp8.size(0)
                out = torch.empty(
                    dim_size, dtype=proj_out_pttype, device=inputmat_total.device
                )

            _ = fp8_gemm(
                weight_fp8._data,
                weight_fp8._scale_inv,
                0,
                weight_fp8._fp8_dtype,
                (
                    inputmat_total._data
                    if isinstance(inputmat_total, Float8Tensor)
                    else inputmat_total
                ),
                inputmat_scale_inv,
                0,
                fp8_dtype_forward,
                proj_out_pttype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
                use_split_accumulator=_2X_ACC_FPROP,
                out=out,
                ub_algo=ub_algo if ub_overlap_rs else None,
                ub=ub_obj if ub_overlap_rs else None,
                extra_output_tensor=rs_out if ub_overlap_rs else None,
                out_index=proj_out_index,
                fp8_meta_tensor=meta_tensor,
                D_dtype=proj_out_tetype,
            )
            if fp8_output:
                out = Float8Tensor(
                    data=out,
                    fp8_meta=fp8_meta,
                    fp8_meta_forward=True,
                    fp8_meta_index=tex.FP8FwdTensors.GEMM1_OUTPUT,
                    fp8_dtype=fp8_dtype_forward,
                    dtype=activation_dtype,
                )
        else:
            # Cast for native AMP
            weight = cast_if_needed(weight, activation_dtype)
            bias = cast_if_needed(bias, activation_dtype) if use_bias else bias

            # (TODO): make fp8_calibration setup right later
            assert (
                not fp8_calibration
            ), "fp8_calibration is not supported now in _A2ALinear"
            if fp8_calibration:
                # amax of input
                amin, amax = inputmat_total.aminmax()
                fp8_meta["scaling_fwd"].amax_history[0][
                    tex.FP8FwdTensors.GEMM1_INPUT
                ] = torch.max(-amin, amax).float()
                # amax of weight
                amin, amax = weight.aminmax()
                fp8_meta["scaling_fwd"].amax_history[0][
                    tex.FP8FwdTensors.GEMM1_WEIGHT
                ] = torch.max(-amin, amax).float()

            ub_algo = None
            extra_output_tensor = None
            if ub_overlap_rs and parallel_mode == "row":
                out = ub_obj.get_ubuf_output(1)
                dim_size = list(inputmat_total.size())
                dim_size[0] = dim_size[0] // get_distributed_world_size(tp_group)
                dim_size[1] = weight.size(0)
                rs_out = torch.empty(
                    dim_size, dtype=activation_dtype, device=inputmat_total.device
                )
                extra_output_tensor = rs_out
                if ub_obj.is_p2p_overlap():
                    ub_algo = tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS_P2P
                else:
                    ub_algo = tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS
            else:
                dim_size = list(inputmat_total.size())
                dim_size[1] = weight.size(0)
                if a2a_ag_overlap and parallel_mode == "column":
                    dim_size[0] = dim_size[0] * tp_size
                elif rs_a2a_overlap and parallel_mode == "row":
                    dim_size[0] = dim_size[0] // tp_size
                out = torch.empty(
                    dim_size, dtype=activation_dtype, device=inputmat_total.device
                )

            if ub_overlap_ag and parallel_mode == "column":
                if ub_obj.is_atomic_gemm():
                    ub_algo = tex.UbufOverlapAlgo.ATOMIC_GEMM_AG_P2P
                else:
                    ub_algo = tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG_P2P
                extra_output_tensor = torch.empty_like(inputmat)
                inputmat_no_fp8 = extra_output_tensor

            if a2a_ag_overlap and parallel_mode == "column":
                if moe_ring_exchange:
                    _, inputmat_no_fp8, _ = ring_exchange_overlap_ag_aggregate(
                        weight=weight,
                        input=inputmat_total,
                        activation_dtype=activation_dtype,
                        out=out,
                        ub_algo=ub_algo,
                        ub_obj=ub_obj,
                        extra_output_tensor=extra_output_tensor,
                        ep_aggregate=ep_aggregate_size,
                        tp_aggregate=None,
                        ep_group=ep_group,
                        ep_size=ep_size,
                        tp_group=tp_group,
                        tp_size=tp_size,
                        external_streams=ep_streams,
                    )
                    # _, inputmat_no_fp8,_ = ring_exchange_overlap_ag(
                    #             weight=weight,
                    #             input=inputmat_total,
                    #             activation_dtype=activation_dtype,
                    #             out=out,
                    #             ub_algo=ub_algo,
                    #             ub_obj=ub_obj,
                    #             extra_output_tensor=extra_output_tensor,
                    #             ep_group=ep_group,
                    #             ep_size=ep_size,
                    #             tp_group=tp_group,
                    #             tp_size=tp_size,
                    #             external_streams=ep_streams,
                    #         )

                elif moe_pipeline_split:
                    _, inputmat_no_fp8, _ = pipeline_split_ag(
                        weight=weight,
                        input=inputmat_total,
                        activation_dtype=activation_dtype,
                        out=out,
                        ub_algo=ub_algo,
                        ub_obj=ub_obj,
                        extra_output_tensor=extra_output_tensor,
                        num_pipeline_stage=num_pipeline_stage,
                        ep_group=ep_group,
                        ep_size=ep_size,
                        tp_group=tp_group,
                        tp_size=tp_size,
                        external_streams=ep_streams,
                    )

            elif rs_a2a_overlap and parallel_mode == "row":
                if moe_ring_exchange:
                    func = (
                        ring_exchange_overlap_rs_tp1
                        if tp_size == 1
                        else ring_exchange_overlap_rs
                    )
                    _ = func(
                        weight=weight,
                        input=inputmat_total,
                        activation_dtype=activation_dtype,
                        out=out,
                        ub_algo=ub_algo,
                        ub_obj=ub_obj,
                        extra_output_tensor=extra_output_tensor,
                        ep_aggregate2=False,
                        tp_aggregate2=False,
                        ep_group=ep_group,
                        ep_size=ep_size,
                        tp_group=tp_group,
                        tp_size=tp_size,
                        external_streams=ep_streams,
                    )
                elif moe_pipeline_split:
                    _ = pipeline_split_rs(
                        weight=weight,
                        input=inputmat_total,
                        activation_dtype=activation_dtype,
                        out=out,
                        ub_algo=ub_algo,
                        ub_obj=ub_obj,
                        extra_output_tensor=extra_output_tensor,
                        num_pipeline_stage=num_pipeline_stage,
                        ep_group=ep_group,
                        ep_size=ep_size,
                        tp_group=tp_group,
                        tp_size=tp_size,
                        external_streams=ep_streams,
                    )

            else:
                _ = gemm(
                    weight,
                    inputmat_total,
                    activation_dtype,
                    get_workspace(),
                    bias=bias,
                    use_bias=use_bias,
                    out=out,
                    ub_algo=ub_algo,
                    ub=ub_obj,
                    extra_output_tensor=extra_output_tensor,
                )

        if is_grad_enabled:
            saved_inputmat = None
            saved_inputmat_t = None
            if weight.requires_grad:
                if fp8 and not fp8_meta["recipe"].override_linear_precision.wgrad:
                    if inputmat_t is None:
                        saved_inputmat = inputmat
                    else:
                        saved_inputmat_t = inputmat_t
                        if cpu_offloading:
                            saved_inputmat_t.activation_offloading = True
                else:
                    saved_inputmat = inputmat_no_fp8

                if cpu_offloading:
                    if fp8 and weight_fp8 is not None:
                        weight_fp8.weight_offloading = True
                    weight.weight_offloading = True

                    if saved_inputmat is not None:
                        saved_inputmat.activation_offloading = True

            # Scatter intermediate/activation tensors saved for the backward pass
            # NOTE: FSDP sharding is not valid for models initialized with primary Fp8 weights
            ctx.fsdp_group = fsdp_group
            ctx.fsdp_shapes = _fsdp_scatter_tensors(
                fsdp_group,
                saved_inputmat,  # None if fp8 == False
                saved_inputmat_t,  # None if fp8 == False AND not is_grad_enabled
                weight_fp8 if fp8 and not isinstance(weight, Float8Tensor) else None,
            )

            ctx.save_for_backward(
                saved_inputmat,
                saved_inputmat_t,
                inputmat_scale_inv,
                weight,
                weight_fp8,
                weight.main_grad
                if cpu_offloading and fuse_wgrad_accumulation
                else None,
            )

            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_meta = fp8_meta
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = use_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp.shape
            ctx.parallel_mode = parallel_mode
            ctx.tp_group = tp_group
            ctx.ub_overlap_ag = ub_overlap_ag and (parallel_mode == "row")
            ctx.ub_name = ub_name
            ctx.tp_size = tp_size
            ctx.requires_dgrad = inp.requires_grad
            ctx.is_input_fp8 = is_input_fp8
            ctx.reduce_and_update_bwd_fp8_tensors = False
            if ctx.fp8 and requires_grad(inp, weight, bias):
                ctx.reduce_and_update_bwd_fp8_tensors = (
                    ctx.reduce_and_update_bwd_fp8_tensors
                    or FP8GlobalStateManager.is_first_fp8_module()
                )
            ctx.ep_group = ep_group
            ctx.ep_size = ep_size
            ctx.moe_use_a2alinear = moe_use_a2alinear
            ctx.moe_ring_exchange = moe_ring_exchange
            ctx.moe_pipeline_split = moe_pipeline_split
            ctx.moe_num_pipeline_stage = moe_num_pipeline_stage
            ctx.ep_aggregate_size = ep_aggregate_size
            ctx.ep_streams = ep_streams

        # Row Parallel Linear
        if ub_overlap_rs and parallel_mode == "row":
            out = rs_out
        elif parallel_mode == "row" and sequence_parallel:
            # no UB based overlap
            if not moe_pipeline_split:
                out, _ = reduce_scatter_along_first_dim(out, tp_group)
        elif parallel_mode == "row" and tensor_parallel:
            out, _ = allreduce(out, tp_group)

        if parallel_mode == "row" and moe_use_a2alinear and not rs_a2a_overlap:
            if ep_size != 1:
                out, _ = alltoall(None, out, None, None, ep_group, False,)

        # [*, in_features] -> [*, out_features] except first dimension changes for S
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:

        disable_a2a_ag_overlap = False
        disable_rs_ag_bulk_overlap = False
        disable_ag_bulk_overlap = False

        a2a_ag_overlap = (
            (ctx.moe_ring_exchange or ctx.moe_pipeline_split)
            and ctx.parallel_mode == "row"
            and not disable_a2a_ag_overlap
        )

        rs_a2a_bulk_overlap = (
            not disable_rs_ag_bulk_overlap
            and ctx.parallel_mode == "column"
            and (ctx.moe_ring_exchange or ctx.moe_pipeline_split)
        )

        if ctx.moe_ring_exchange or ctx.moe_pipeline_split:
            ctx.ub_overlap_ag = False

        if isinstance(grad_output, Float8Tensor):
            ctx.fp8_meta["scaling_bwd"].scale_inv[
                tex.FP8BwdTensors.GRAD_OUTPUT1
            ] = grad_output._scale_inv

        with torch.cuda.nvtx.range("_Linear_backward"):
            (
                inputmat,
                inputmat_t,
                inputmat_scale_inv,
                weight,
                weight_fp8,
                main_grad,
            ) = ctx.saved_tensors

            # Gather intermediate/activation tensors if needed
            # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves
            _fsdp_gather_tensors(
                ctx.fsdp_group,
                ctx.fsdp_shapes,
                inputmat,
                inputmat_t,
                weight_fp8
                if ctx.fp8 and not isinstance(weight, Float8Tensor)
                else None,
            )

            if ctx.cpu_offloading and ctx.fuse_wgrad_accumulation:
                weight = torch.nn.Parameter(weight, weight.requires_grad)
                weight.main_grad = main_grad

            tp_world_size = get_distributed_world_size(ctx.tp_group)
            ctx.ub_overlap_ag = False if tp_world_size == 1 else ctx.ub_overlap_ag

            if ctx.ub_overlap_ag:
                dim_size = list(grad_output.size())
                dim_size[0] = dim_size[0] * tp_world_size
                ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad")
                if ctx.ub_obj_gradout.is_atomic_gemm():
                    ub_algo = tex.UbufOverlapAlgo.ATOMIC_GEMM_AG_P2P
                else:
                    ub_algo = tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG_P2P

            if (
                ctx.parallel_mode == "row"
                and ctx.moe_use_a2alinear
                and not a2a_ag_overlap
            ):
                grad_output, _ = alltoall(
                    None,
                    grad_output,
                    None,  # input_splits
                    None,  # output_splits
                    ctx.ep_group,
                    False,  # synchronous operation
                )

            (
                grad_output,
                grad_output_c,
                grad_output_t,
                grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx,
                grad_output,
                ctx.parallel_mode == "row",
                a2a_ag_overlap=a2a_ag_overlap,
            )

            # Column Parallel Linear
            # Overlap input AG with dgrad
            inputmat_total = None
            inputmat_t_total = None
            handle = None
            if (
                weight.requires_grad
                and ctx.parallel_mode == "column"
                and ctx.sequence_parallel
            ):
                if not ctx.moe_pipeline_split:
                    inputmat_total, handle = gather_along_first_dim(
                        inputmat, ctx.tp_group, async_op=ctx.requires_dgrad
                    )
            else:
                inputmat_total = inputmat
                inputmat_t_total = inputmat_t

            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            if ctx.fp8:
                fp8_dtype_forward = get_fp8_te_dtype(
                    ctx.fp8_meta["recipe"], fprop_tensor=True
                )
                fp8_dtype_backward = get_fp8_te_dtype(
                    ctx.fp8_meta["recipe"], fprop_tensor=False
                )

            if ctx.requires_dgrad:
                if ctx.fp8:
                    if ctx.is_input_fp8:
                        out_index, meta_tensor, output_te_dtype, output_dtype = (
                            tex.FP8BwdTensors.GRAD_INPUT1,
                            ctx.fp8_meta["scaling_bwd"],
                            fp8_dtype_backward,
                            torch.uint8,
                        )
                    else:
                        out_index, meta_tensor, output_te_dtype, output_dtype = (
                            None,
                            None,
                            None,
                            ctx.activation_dtype,
                        )
                    dgrad, _ = fp8_gemm(
                        weight_fp8.transpose_2d(),
                        weight_fp8._scale_inv,
                        0,
                        weight_fp8._fp8_dtype,
                        grad_output_c,
                        ctx.fp8_meta["scaling_bwd"].scale_inv,
                        tex.FP8BwdTensors.GRAD_OUTPUT1,
                        fp8_dtype_backward,
                        output_dtype,
                        get_workspace(),
                        use_split_accumulator=_2X_ACC_DGRAD,
                        ub_algo=ub_algo if ctx.ub_overlap_ag else None,
                        ub=ctx.ub_obj_gradout if ctx.ub_overlap_ag else None,
                        out_index=out_index,
                        fp8_meta_tensor=meta_tensor,
                        D_dtype=output_te_dtype,
                    )
                    if output_dtype == torch.uint8:
                        dgrad = Float8Tensor(
                            data=dgrad,
                            fp8_meta=ctx.fp8_meta,
                            fp8_meta_forward=False,
                            fp8_meta_index=tex.FP8BwdTensors.GRAD_INPUT1,
                            fp8_dtype=fp8_dtype_backward,
                            dtype=ctx.activation_dtype,
                        )
                else:
                    if a2a_ag_overlap and ctx.parallel_mode == "row":
                        dgrad = torch.empty(
                            (grad_output.size(0) * ctx.tp_size, weight.size(1)),
                            dtype=ctx.activation_dtype,
                            device=grad_output.device,
                        )
                        if ctx.moe_ring_exchange:
                            _, _, grad_output = ring_exchange_overlap_ag_aggregate(
                                weight=weight,
                                input=grad_output,
                                activation_dtype=ctx.activation_dtype,
                                out=dgrad,
                                ub_algo=None,
                                ub_obj=None,
                                extra_output_tensor=None,
                                ep_aggregate=ctx.ep_aggregate_size,
                                tp_aggregate=None,
                                ep_group=ctx.ep_group,
                                ep_size=ctx.ep_size,
                                tp_group=ctx.tp_group,
                                tp_size=ctx.tp_size,
                                layout="NN",
                                grad=True,
                                external_streams=ctx.ep_streams,
                            )
                            # _, _, grad_output = ring_exchange_overlap_ag(
                            #         weight=weight,
                            #         input=grad_output,
                            #         activation_dtype=ctx.activation_dtype,
                            #         out=dgrad,
                            #         ub_algo=None,
                            #         ub_obj=None,
                            #         extra_output_tensor=None,
                            #         ep_group=ctx.ep_group,
                            #         ep_size=ctx.ep_size,
                            #         tp_group=ctx.tp_group,
                            #         tp_size=ctx.tp_size,
                            #         layout="NN",
                            #         grad=True,
                            #         external_streams=ctx.ep_streams,
                            #     )
                        elif ctx.moe_pipeline_split:
                            _, _, grad_output = pipeline_split_ag(
                                weight=weight,
                                input=grad_output,
                                activation_dtype=ctx.activation_dtype,
                                out=dgrad,
                                ub_algo=None,
                                ub_obj=None,
                                extra_output_tensor=None,
                                num_pipeline_stage=ctx.moe_num_pipeline_stage,
                                ep_group=ctx.ep_group,
                                ep_size=ctx.ep_size,
                                tp_group=ctx.tp_group,
                                tp_size=ctx.tp_size,
                                layout="NN",
                                grad=True,
                                external_streams=ctx.ep_streams,
                            )
                    elif ctx.moe_pipeline_split and ctx.parallel_mode == "column":
                        dgrad = torch.empty(
                            (grad_output.size(0), weight.size(1)),
                            dtype=ctx.activation_dtype,
                            device=grad_output.device,
                        )
                        inputmat_total = pipeline_split_bulk_ag(
                            weight=weight,
                            input=grad_output,
                            activation_dtype=ctx.activation_dtype,
                            out=dgrad,
                            ub_algo=None,
                            ub_obj=None,
                            extra_output_tensor=None,
                            num_pipeline_stage=ctx.moe_num_pipeline_stage,
                            ep_group=ctx.ep_group,
                            ep_size=ctx.ep_size,
                            tp_group=ctx.tp_group,
                            tp_size=ctx.tp_size,
                            layout="NN",
                            grad=True,
                            ag_in=inputmat,
                            external_streams=ctx.ep_streams,
                        )

                    else:
                        dgrad, _, _ = gemm(
                            weight,
                            grad_output,
                            ctx.activation_dtype,
                            get_workspace(),
                            layout="NN",
                            grad=True,
                            ub_algo=(
                                tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG_P2P
                                if ctx.ub_overlap_ag
                                else None
                            ),
                            ub=ctx.ub_obj_gradout if ctx.ub_overlap_ag else None,
                        )

                # Overlap dgrad-RS/AR with wgrad
                if ctx.parallel_mode == "column" and ctx.sequence_parallel:
                    if handle is not None:
                        handle.wait()
                    if not rs_a2a_bulk_overlap:
                        dgrad, handle = reduce_scatter_along_first_dim(
                            dgrad, ctx.tp_group, async_op=True
                        )
                    else:
                        handle = None
                elif ctx.parallel_mode == "column" and ctx.tensor_parallel:
                    dgrad, handle = allreduce(dgrad, ctx.tp_group, async_op=True)
            if weight.requires_grad:
                if ctx.fp8:
                    # WGRAD
                    if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                        if ctx.ub_overlap_ag:
                            if isinstance(grad_output_c, Float8Tensor):
                                grad_output_t = grad_output_c.transpose_2d()
                            else:
                                grad_output_t = tex.fp8_transpose(
                                    grad_output_c, fp8_dtype_backward
                                )
                        if inputmat_t_total is None:
                            if isinstance(inputmat_total, Float8Tensor):
                                inputmat_t_total = inputmat_total.transpose_2d()
                            else:
                                inputmat_t_total = tex.fp8_transpose(
                                    inputmat_total, fp8_dtype_backward
                                )
                        wgrad, _ = fp8_gemm(
                            (
                                inputmat_t_total._data
                                if isinstance(inputmat_t_total, Float8Tensor)
                                else inputmat_t_total
                            ),
                            inputmat_scale_inv,
                            0,
                            fp8_dtype_forward,
                            grad_output_t,
                            ctx.fp8_meta["scaling_bwd"].scale_inv,
                            tex.FP8BwdTensors.GRAD_OUTPUT1,
                            fp8_dtype_backward,
                            ctx.activation_dtype,
                            get_workspace(),
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            out=weight.main_grad
                            if ctx.fuse_wgrad_accumulation
                            else None,
                            use_split_accumulator=_2X_ACC_WGRAD,
                        )
                    else:
                        wgrad, _, _ = gemm(
                            inputmat_total,
                            grad_output,
                            ctx.activation_dtype,
                            get_workspace(),
                            layout="NT",
                            grad=True,
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            out=weight.main_grad
                            if ctx.fuse_wgrad_accumulation
                            else None,
                        )
                else:
                    if rs_a2a_bulk_overlap and ctx.parallel_mode == "column":
                        if ctx.moe_ring_exchange:
                            wgrad, _, dgrad = rs_a2a_bulk_overlap_gemm(
                                inputmat_total,
                                grad_output,
                                ctx.activation_dtype,
                                ctx.ep_group,
                                ctx.ep_size,
                                ctx.tp_group,
                                ctx.tp_size,
                                layout="NT",
                                grad=True,
                                rs_in=dgrad,
                                accumulate_wgrad_into_param_main_grad=accumulate_wgrad_into_param_main_grad,
                                out=weight.main_grad
                                if ctx.fuse_wgrad_accumulation
                                else None,
                                external_streams=ctx.ep_streams,
                            )
                        elif ctx.moe_pipeline_split:
                            wgrad, _, dgrad = pipeline_split_bulk_rs_a2a(
                                inputmat_total,
                                grad_output,
                                ctx.activation_dtype,
                                ctx.moe_num_pipeline_stage,
                                ctx.ep_group,
                                ctx.ep_size,
                                ctx.tp_group,
                                ctx.tp_size,
                                layout="NT",
                                grad=True,
                                rs_in=dgrad,
                                accumulate_wgrad_into_param_main_grad=accumulate_wgrad_into_param_main_grad,
                                out=weight.main_grad
                                if ctx.fuse_wgrad_accumulation
                                else None,
                                external_streams=ctx.ep_streams,
                            )
                    else:
                        # WGRAD
                        wgrad, grad_bias, _ = gemm(
                            inputmat_total,
                            grad_output,
                            ctx.activation_dtype,
                            get_workspace(),
                            layout="NT",
                            grad=True,
                            use_bias=ctx.use_bias,
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            out=weight.main_grad
                            if ctx.fuse_wgrad_accumulation
                            else None,
                        )

                # Deallocate input tensor
                clear_tensor_data(inputmat_total)
                clear_tensor_data(inputmat_t_total)

            # Column Parallel Linear
            if (
                ctx.parallel_mode == "column"
                and ctx.tensor_parallel
                and handle is not None
            ):
                handle.wait()

            # A2A at dgrad RS output for column gemm
            if (
                ctx.parallel_mode == "column"
                and ctx.moe_use_a2alinear
                and not rs_a2a_bulk_overlap
            ):
                if ctx.ep_size != 1:
                    dgrad, _ = alltoall(
                        None,
                        dgrad,
                        None,  # input_splits
                        None,  # output_splits
                        ctx.ep_group,
                        False,  # synchronous operation
                    )

            if not ctx.use_bias:
                grad_bias = None

        if weight.requires_grad:
            # Handle custom DDP from mcore.
            if ctx.fuse_wgrad_accumulation and hasattr(
                weight, "grad_added_to_main_grad"
            ):
                weight.grad_added_to_main_grad = True
                if getattr(weight, "zero_out_wgrad", False):
                    wgrad = torch.zeros(
                        weight.main_grad.shape,
                        dtype=weight.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    wgrad = torch.empty(
                        weight.main_grad.shape,
                        dtype=weight.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
            elif ctx.fuse_wgrad_accumulation:
                wgrad = None
        else:
            wgrad = None

        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
        # Scatter fp8 weight buffers
        if ctx.fp8 and not isinstance(weight, Float8Tensor):
            _fsdp_scatter_tensors(ctx.fsdp_group, weight_fp8)
        return (
            wgrad,
            None,  # weight_fp8
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            grad_bias,
            None,  # use_bias
            None,  # is_first_microbatch
            None,  # fp8
            None,  # fp8_calibration
            None,  # fp8_meta
            None,  # fuse_wgrad_accumulation
            None,  # cpu_offloading
            None,  # tp_group
            None,  # tp_size
            None,  # sequence_parallel
            None,  # tensor_parallel
            None,  # activation_dtype
            None,  # parallel_mode
            None,  # is_grad_enabled
            None,  # ub_overlap_rs
            None,  # ub_overlap_ag
            None,  # ub_name
            None,  # fp8_output
            None,  # fsdp_group
            None,  # ep_group
            None,  # ep_size
            None,  # moe_use_a2alinear
            None,  # moe_ring_exchange
            None,  # moe_pipeline_split
            None,  # moe_num_pipeline_stage
            None,  # ep_streams
        )
