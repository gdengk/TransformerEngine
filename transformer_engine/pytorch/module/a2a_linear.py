# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
from typing import Any, Callable, Dict, Optional, Tuple, Union

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
    reverse_order,
    send_tensor,
    send_dst,
    recv_tensor,
    recv_src,
    process_group,
    batch_p2p_comm,
):
    send_recv_ops = []
    if not reverse_order:
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


def a2a_p2p_with_step(step, send_tensor, recv_tensor, process_group, batch_p2p_comm):
    # step has to be a positive num
    send_recv_ops = []
    pg_world_size = torch.distributed.get_world_size(process_group)
    local_rank = torch.distributed.get_rank(process_group)
    gcd = math.gcd(pg_world_size, step)
    reverse_order = (local_rank // gcd) % 2 == 1

    send_dst = torch.distributed.get_global_rank(
        process_group, (local_rank + step + pg_world_size) % pg_world_size
    )
    recv_src = torch.distributed.get_global_rank(
        process_group, (local_rank - step + pg_world_size) % pg_world_size
    )

    if not reverse_order:
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

    if batch_p2p_comm:
        send_recv_reqs = torch.distributed.batch_isend_irecv(send_recv_ops)
    else:
        send_recv_reqs = send_recv_ops
    return send_recv_reqs


def multi_step_p2p(
    step_list,
    send_tensor_list,
    recv_tensor_list,
    sub_pg_size,
    process_group,
    batch_p2p_comm,
):

    local_rank = torch.distributed.get_rank(process_group)
    sub_pg_offset = local_rank // sub_pg_size
    sub_pg_rank = local_rank % sub_pg_size

    all_send_recv_ops = []

    for step, send_tensor, recv_tensor in zip(
        step_list, send_tensor_list, recv_tensor_list
    ):
        gcd = math.gcd(sub_pg_size, step)
        reverse_order = (sub_pg_rank // gcd) % 2 == 1
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
                reverse_order,
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
    chunked_input,
    comm_buf,
    aggregate_size,
    comm_buf_nonag_offset,
    process_group,
    batch_p2p_comm,
):
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
    reverse_order = (chunk_idx // gcd) % 2 == 1

    for i in range(aggregate_size):

        dst_rank = (
            dst_chunk_idx * aggregate_size + (i + sub_chunk_rank) % aggregate_size
        )
        src_rank = (
            src_chunk_idx * aggregate_size
            + (sub_chunk_rank - i + aggregate_size) % aggregate_size
        )

        send_tensor = chunked_input[dst_rank]
        send_dst = torch.distributed.get_global_rank(process_group, dst_rank)
        recv_tensor = comm_buf[src_rank + comm_buf_nonag_offset]
        recv_src = torch.distributed.get_global_rank(process_group, src_rank)

        send_recv_ops.extend(
            get_send_recv_ops(
                reverse_order,
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
    out,  # likely not need it,
    ub_algo,
    ub_obj,
    extra_output_tensor,
    ep_aggregate2,
    tp_aggregate2,  # ag aggregate will affect data ordering
    ep_group,
    ep_size,
    tp_group,
    tp_size,
    layout="TN",
    grad=False,
):
    # multistream ring exchange
    # disable a2a+ag outside this loop
    # disable rs+ag outside this loop

    # expose this later
    batch_p2p_comm = True

    ep_local_rank = torch.distributed.get_rank(ep_group) if ep_size != 1 else None
    tp_local_rank = torch.distributed.get_rank(tp_group) if tp_size != 1 else 0

    # disable aggregate2 when no tp/ep group
    ep_aggregate2 = False if ep_size == 1 else ep_aggregate2
    tp_aggregate2 = False if tp_size == 1 else tp_aggregate2

    ep_chunksize = ep_size // 2 if ep_aggregate2 else ep_size
    tp_chunksize = tp_size // 2 if tp_aggregate2 else tp_size

    # [torch.empty_like(input) for _ in range(ep_chunksize * tp_chunksize)]
    # comm_buf = [torch.empty_like(input) for _ in range(ep_chunksize * tp_chunksize)]
    ep_reqs = []
    tp_reqs = []

    # two streams for now
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    # chunk input as view:
    chunked_input = torch.chunk(input, chunks=ep_size, dim=0)
    # consider ag only at this time
    tmp_buf = torch.empty(
        (input.shape[0] * tp_size),
        input.shape[1],
        dtype=input.dtype,
        device=input.device,
    )
    comm_buf = torch.chunk(tmp_buf, chunks=ep_chunksize * tp_chunksize, dim=0)
    # must assign out
    assert out is not None, "must assign out before ring_exchange_overlap function"
    chunked_out = torch.chunk(out, chunks=ep_chunksize * tp_chunksize, dim=0)
    assert ub_obj is None, "ring_exchange_overlap does not support UB for now"
    assert (
        extra_output_tensor is None
    ), "ring_exchange_overlap does not support extra_output_tensor from UB"

    # make sure it copied:
    comm_buf_offset = tp_local_rank * ep_chunksize
    comm_buf[ep_local_rank + comm_buf_offset].copy_(chunked_input[ep_local_rank])

    cublasworkspace = get_multi_stream_cublas_workspace()

    for s in streams:
        s.wait_stream(torch.cuda.current_stream())

    for i in range(ep_chunksize):
        src_ep_i = (ep_local_rank + i + 1) % ep_size
        dst_ep_i = (ep_local_rank - 1 - i + ep_size) % ep_size
        cur_ep_i = (ep_local_rank - i + ep_size) % ep_size
        for j in range(tp_chunksize):
            streamid = (i * tp_chunksize + j) % 2
            with torch.cuda.stream(streams[streamid]):
                if j == 0:
                    # data dependency check before starting AG
                    for req in ep_reqs:
                        req.wait()

                    # launch the a2a before AG starts
                    if i < ep_chunksize - 1:
                        if not ep_aggregate2:
                            dst_comm_buf_i = dst_ep_i + tp_local_rank * ep_chunksize
                            ep_reqs = a2a_p2p_with_step(
                                i + 1,
                                chunked_input[src_ep_i],
                                comm_buf[dst_comm_buf_i],
                                ep_group,
                                batch_p2p_comm,
                            )
                        else:
                            # (TODO)
                            pass

                        # if i == 0:
                        #     tp_in = chunked_input[ep_local_rank]
                        # else:
                        # dst_comm_buf_i = cur_ep_i + tp_local_rank * ep_chunksize
                        # tp_in = comm_buf[dst_comm_buf_i]

                        if tp_aggregate2:
                            # [todo] launch the first tp communication
                            pass

                # tp comm starts
                for req in tp_reqs:
                    req.wait()

                cur_tp_i = (tp_local_rank - j + tp_size) % tp_size
                dst_tp_i = (tp_local_rank - j - 1 + tp_size) % tp_size
                src_comm_buf_i = cur_tp_i * ep_chunksize + cur_ep_i
                if j < tp_chunksize - 1:
                    if not tp_aggregate2:
                        dst_comm_buf_i = dst_tp_i * ep_chunksize + cur_ep_i
                        tp_reqs = a2a_p2p_with_step(
                            1,
                            comm_buf[src_comm_buf_i],
                            comm_buf[dst_comm_buf_i],
                            tp_group,
                            batch_p2p_comm,
                        )
                    else:
                        # (TODO)
                        pass  # for now
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

                # in the last loop, tp_dst_i keep the same as prev value but not used.
                # tp_in = comm_buf[dst_comm_buf_i]

    for s in streams:
        torch.cuda.current_stream().wait_stream(s)

    # (TODO): if needs grad
    # need to hide this D2D with gemm
    # copy current data to coresponding buf
    # (TODO) we might have a way to get rid of this D2D and make it async
    # comm_buf_offset = tp_local_rank * ep_chunksize
    # comm_buf[ep_local_rank + comm_buf_offset].copy_(chunked_input[ep_local_rank])
    dim_size = comm_buf[0].size(0)
    a2a_out = tmp_buf[
        comm_buf_offset * dim_size : (comm_buf_offset + ep_chunksize) * dim_size
    ]

    return out, a2a_out


def ring_exchange_overlap_ag_aggregate(
    weight,
    input,
    activation_dtype,
    out,  # likely not need it,
    ub_algo,
    ub_obj,
    extra_output_tensor,
    ep_aggregate,
    tp_aggregate,  # ag aggregate will affect data ordering
    ep_group,
    ep_size,
    tp_group,
    tp_size,
    layout="TN",
    grad=False,
):
    # multistream ring exchange
    # disable a2a+ag outside this loop
    # disable rs+ag outside this loop

    # expose this later
    batch_p2p_comm = True

    ep_local_rank = torch.distributed.get_rank(ep_group) if ep_size != 1 else 0
    tp_local_rank = torch.distributed.get_rank(tp_group) if tp_size != 1 else 0

    # disable aggregate2 when no tp/ep group
    if ep_aggregate is None or ep_size == 1:
        ep_aggregate = 1

    # assert ep_aggregate should be an int and should be at least half of ep_size

    ep_chunksize = ep_size // ep_aggregate
    # tp_chunksize = tp_size // 2 if tp_aggregate2 else tp_size
    tp_chunksize = tp_size
    assert tp_aggregate is None, "Don't supoort tp_aggregate for now"

    ep_reqs = []
    tp_reqs = []

    # two streams for now
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    # chunk input as view:
    chunked_input = torch.chunk(input, chunks=ep_size, dim=0)
    # consider ag only at this time
    comm_full_buf = torch.empty(
        (input.shape[0] * tp_size),
        input.shape[1],
        dtype=input.dtype,
        device=input.device,
    )
    comm_buf_ag_view = torch.chunk(
        comm_full_buf, chunks=ep_chunksize * tp_chunksize, dim=0
    )
    comm_buf_nonag_view = torch.chunk(comm_full_buf, chunks=ep_size * tp_size, dim=0)
    # must assign out
    assert out is not None, "must assign out before ring_exchange_overlap function"
    chunked_out = torch.chunk(out, chunks=ep_chunksize * tp_chunksize, dim=0)
    assert ub_obj is None, "ring_exchange_overlap does not support UB for now"
    assert (
        extra_output_tensor is None
    ), "ring_exchange_overlap does not support extra_output_tensor from UB"

    # make sure it copied:
    comm_buf_nonag_offset = tp_local_rank * ep_size
    comm_buf_ag_offset = tp_local_rank * ep_chunksize

    ep_chunk_idx = ep_local_rank // ep_aggregate

    cublasworkspace = get_multi_stream_cublas_workspace()

    for s in streams:
        s.wait_stream(torch.cuda.current_stream())

    # preprocess aggregate input
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

        ep_reqs = multi_step_p2p(
            preprocess_step_list,
            preprocess_send_tensor_list,
            preprocess_recv_tensor_list,
            ep_aggregate,
            ep_group,
            batch_p2p_comm,
        )

    # local D2D into buf
    comm_buf_nonag_view[ep_local_rank + comm_buf_nonag_offset].copy_(
        chunked_input[ep_local_rank]
    )

    for i in range(ep_chunksize):
        cur_ep_i = (ep_chunk_idx - i + ep_chunksize) % ep_chunksize
        for j in range(tp_chunksize):
            streamid = (i * tp_chunksize + j) % 2
            with torch.cuda.stream(streams[streamid]):
                if j == 0:
                    # data dependency check before starting AG
                    for req in ep_reqs:
                        req.wait()

                    # launch the a2a before AG starts
                    if i < ep_chunksize - 1:
                        aggregate_p2p(
                            i + 1,
                            chunked_input,
                            comm_buf_nonag_view,
                            ep_aggregate,
                            comm_buf_nonag_offset,
                            ep_group,
                            batch_p2p_comm,
                        )

                # tp comm starts
                for req in tp_reqs:
                    req.wait()

                cur_tp_i = (tp_local_rank - j + tp_size) % tp_size
                dst_tp_i = (tp_local_rank - j - 1 + tp_size) % tp_size
                src_comm_buf_i = cur_tp_i * ep_chunksize + cur_ep_i
                if j < tp_chunksize - 1:
                    dst_comm_buf_i = dst_tp_i * ep_chunksize + cur_ep_i
                    tp_reqs = a2a_p2p_with_step(
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

                # in the last loop, tp_dst_i keep the same as prev value but not used.
                # tp_in = comm_buf[dst_comm_buf_i]

    for s in streams:
        torch.cuda.current_stream().wait_stream(s)

    # (TODO): Can we make async D2D
    dim_size = comm_buf_nonag_view[0].size(0)
    a2a_out = comm_full_buf[
        comm_buf_nonag_offset * dim_size : (comm_buf_nonag_offset + ep_size) * dim_size
    ]

    return out, a2a_out


def ring_exchange_overlap_rs(
    weight,
    input,
    activation_dtype,
    out,  # likely not need it,
    ub_algo,
    ub_obj,
    extra_output_tensor,
    ep_aggregate2,
    tp_aggregate2,
    ep_group,
    ep_size,
    tp_group,
    tp_size,
):
    batch_p2p_comm = True
    ep_local_rank = torch.distributed.get_rank(ep_group) if ep_size != 1 else None

    # disable aggregate2 when no tp/ep group
    ep_aggregate2 = False if ep_size == 1 else ep_aggregate2
    ep_chunksize = ep_size // 2 if ep_aggregate2 else ep_size

    ep_reqs = []

    chunked_input = torch.chunk(input, chunks=ep_size * tp_size, dim=0)
    # a2a output
    if tp_size != 1:
        chunked_out = torch.chunk(extra_output_tensor, chunks=ep_size, dim=0)
    else:
        chunked_out = torch.chunk(out, chunks=ep_size, dim=0)

    rs_out = [torch.empty_like(chunked_out[0]) for _ in range(ep_size - 1)]

    cublasworkspace = get_workspace()

    for i in range(ep_chunksize):
        cur_ep_i = (ep_local_rank + i + 1) % ep_size
        dst_ep_i = (ep_local_rank - i - 1 + ep_size) % ep_size

        if i == ep_chunksize - 1:
            gemm_rs_out = chunked_out[dst_ep_i]
        else:
            gemm_rs_out = rs_out[i]
        if tp_size == 1:
            gemm_out = gemm_rs_out
            gemm_extra_output_tensor = None
        else:
            gemm_out = ub_obj.get_ubuf_output(1)
            gemm_extra_output_tensor = gemm_rs_out

        gemm_in = torch.cat(chunked_input[cur_ep_i::ep_size], dim=0)
        _ = gemm(
            weight,
            gemm_in,
            activation_dtype,
            cublasworkspace,
            out=gemm_out,
            ub_algo=ub_algo,
            ub=ub_obj,
            extra_output_tensor=gemm_extra_output_tensor,
        )

        if i < ep_chunksize - 1:
            reqs = a2a_p2p_with_step(
                i + 1, rs_out[i], chunked_out[dst_ep_i], ep_group, batch_p2p_comm
            )
            ep_reqs.extend(reqs)

    for req in ep_reqs:
        req.wait()

    return extra_output_tensor


def ring_exchange_overlap_rs_tp1(
    weight,
    input,
    activation_dtype,
    out,  # likely not need it,
    ub_algo,
    ub_obj,
    extra_output_tensor,
    ep_aggregate2,
    tp_aggregate2,
    ep_group,
    ep_size,
    tp_group,
    tp_size,
):
    batch_p2p_comm = True
    assert tp_size == 1, "only support TP1 in this case"
    ep_local_rank = torch.distributed.get_rank(ep_group) if ep_size != 1 else None

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    # disable aggregate2 when no tp/ep group
    ep_aggregate2 = False if ep_size == 1 else ep_aggregate2
    ep_chunksize = ep_size // 2 if ep_aggregate2 else ep_size

    ep_reqs = []

    chunked_input = torch.chunk(input, chunks=ep_size * tp_size, dim=0)
    # a2a output
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
                reqs = a2a_p2p_with_step(
                    i + 1, rs_out[i], chunked_out[dst_ep_i], ep_group, batch_p2p_comm
                )
                ep_reqs.extend(reqs)
    for s in streams:
        torch.cuda.current_stream().wait_stream(s)

    for req in ep_reqs:
        req.wait()

    return out


def rs_a2a_bulk_overlap_gemm_bak1(
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
):

    streams = [torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()]

    if tp_size == 1 or ep_size == 1:
        chunk_size = 1
        chunked_rs_in = [rs_in]
    else:
        chunk_size = 4
        chunked_rs_in = torch.chunk(rs_in, chunks=chunk_size, dim=1)
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

    with torch.cuda.stream(streams[0]):
        if tp_size != 1:
            for i in range(chunk_size):
                rs_handle = torch.distributed._reduce_scatter_base(
                    a2a_in[i],
                    chunked_rs_in[i].contiguous(),
                    group=tp_group,
                    async_op=True,
                )
                rs_handles.append(rs_handle)
        else:
            a2a_in[0] = rs_in
            rs_handles = []

    with torch.cuda.stream(streams[1]):
        if ep_size != 1:
            for i in range(chunk_size):
                if len(rs_handles) != 0:
                    rs_handles[i].wait()
                a2a_handle = torch.distributed.all_to_all_single(
                    a2a_output[i], a2a_in[i], None, None, ep_group, async_op=True,
                )
                a2a_handles.append(a2a_handle)
        else:
            a2a_output[0] = a2a_in[0]
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

    return wgrad, grad_bias, rs_out


def rs_a2a_bulk_overlap_gemm_bak2(
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
):

    streams = [torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()]

    if tp_size == 1 or ep_size == 1:
        chunk_size = 1
        chunked_rs_in = [rs_in]
    else:
        chunk_size = 2
        chunked_rs_in = torch.chunk(rs_in, chunks=chunk_size, dim=1)
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

    with torch.cuda.stream(streams[0]):
        if tp_size != 1:
            for i in range(chunk_size):
                rs_handle = torch.distributed._reduce_scatter_base(
                    a2a_in[i],
                    chunked_rs_in[i].contiguous(),
                    group=tp_group,
                    async_op=True,
                )
                rs_handles.append(rs_handle)
        else:
            a2a_in[0] = rs_in
            rs_handles = []

    with torch.cuda.stream(streams[1]):
        if ep_size != 1:
            for i in range(chunk_size):
                if len(rs_handles) != 0:
                    rs_handles[i].wait()
                a2a_handle = torch.distributed.all_to_all_single(
                    a2a_output[i], a2a_in[i], None, None, ep_group, async_op=True,
                )
                a2a_handles.append(a2a_handle)
        else:
            a2a_output[0] = a2a_in[0]
            a2a_handles = rs_handles

    for a2a_handle in a2a_handles:
        a2a_handle.wait()

    for s in streams:
        torch.cuda.current_stream().wait_stream(s)

    if tp_size == 1 or ep_size == 1:
        rs_out = a2a_output[0]
    else:
        rs_out = torch.cat(a2a_output, dim=1)

    return wgrad, grad_bias, rs_out


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
):

    streams = [torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()]

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
                a2a_in[i] = rs_in
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
                a2a_output[0] = a2a_in[0]
                a2a_handles = rs_handles

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
        moe_alltoall_overlap: bool,
        moe_ring_exchange: bool,
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
        # ub_overlap_ag = False if tp_world_size == 1 else ub_overlap_ag
        ub_overlap_ag = False

        # (TODO: Gao for debug) some control commands
        # here moe_alltoall_overlap means move a2a and ag into TE side, does not necessarily means they are going to be overlap
        a2a_ag_overlap = moe_ring_exchange  # column mode
        rs_a2a_overlap = moe_ring_exchange  # moe_ring_exchange # row mode

        # Fake cmd to control TP UB overlap:
        # ub_overlap_ag = True
        # ub_overlap_rs = True

        ## end of debug cmds

        # get ub bufname:
        if (ub_overlap_ag and parallel_mode == "column") or (
            ub_overlap_rs and parallel_mode == "row"
        ):
            ub_obj = get_ub(ub_name + "_fprop")
        else:
            ub_obj = None

        if (
            moe_alltoall_overlap
            and ep_size != 1
            and not a2a_ag_overlap
            and parallel_mode == "column"
        ):
            # A2A communication - evenly distributed
            # (TODO) if not using UB here, save this inputmat as output tensor
            if ub_overlap_ag:
                # for bf16 high precisoin, the output will be directly from a2a out.
                # otherwise this will be downcast to the ubuf[0] as the output
                a2a_out = ub_obj.get_ubuf_output(0)
            else:
                a2a_out = None

            inputmat, _ = alltoall(
                a2a_out,  # don't assign output for now
                inputmat,
                None,  # input_splits for now
                None,  # output_splits for now
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
                # (TODO: Gao) support fp8 later. Here this `not fp8` is inappropriate
                dim_size = list(inputmat.size())
                dim_size[0] = dim_size[0] * tp_world_size
                inputmat_total = ub_obj.get_ubuf_output(1)
                # gemm_in = ub_obj.get_ubuf_output(0)
            else:
                inputmat_total, _ = gather_along_first_dim(inputmat, tp_group)
        else:
            # if a2a_ag_overlap, A2A and AG will be performed in submodule
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

            # (TODO) remove this later
            assert not fp8_calibration, "fp8_calibration must be false in tmp run"
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
                # ring_exchange_overlap_ag_aggregate
                _, inputmat_no_fp8 = ring_exchange_overlap_ag_aggregate(
                    weight=weight,
                    input=inputmat_total,
                    activation_dtype=activation_dtype,
                    out=out,
                    ub_algo=ub_algo,
                    ub_obj=ub_obj,
                    extra_output_tensor=extra_output_tensor,
                    ep_aggregate=1,
                    tp_aggregate=None,
                    ep_group=ep_group,
                    ep_size=ep_size,
                    tp_group=tp_group,
                    tp_size=tp_size,
                )
            elif rs_a2a_overlap and parallel_mode == "row":
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
            ctx.moe_alltoall_overlap = moe_alltoall_overlap
            ctx.moe_ring_exchange = moe_ring_exchange

        # Row Parallel Linear
        if ub_overlap_rs and parallel_mode == "row":
            out = rs_out
        elif parallel_mode == "row" and sequence_parallel:
            # no ub based overlap
            out, _ = reduce_scatter_along_first_dim(out, tp_group)
        elif parallel_mode == "row" and tensor_parallel:
            out, _ = allreduce(out, tp_group)

        # Make sure the saved input for bwd is correct basically something after a2a before ag
        if parallel_mode == "row" and moe_alltoall_overlap and not rs_a2a_overlap:
            if ep_size != 1:
                out, _ = alltoall(None, out, None, None, ep_group, False,)

        # [*, in_features] -> [*, out_features] except first dimension changes for S
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:

        # (TODO: Gao) my backward control port:
        rs_a2a_overlap = (
            False  # get rid of max_num_device_connections = 1 or green context
        )
        a2a_ag_overlap = (
            ctx.moe_ring_exchange
        )  # ctx.moe_ring_exchange #ctx.moe_ring_exchange # might easier to do
        ctx.a2a_ag_overlap = a2a_ag_overlap
        rs_a2a_bulk_overlap = True
        rs_a2a_bulk_overlap = (
            rs_a2a_bulk_overlap
            and ctx.parallel_mode == "column"
            and ctx.moe_ring_exchange
        )

        # end of debug control session

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

            # A2A communication:
            # Row parallel gemm, FC2 : a2a_ag_dgrad overlap
            # Col parallel gemm, FC1 : a2a_rs overlap and bulk overlap with FC2 wgemm

            if (
                ctx.parallel_mode == "row"
                and ctx.moe_alltoall_overlap
                and not a2a_ag_overlap
            ):
                grad_output, _ = alltoall(
                    None,  # don't assign output for now
                    grad_output,
                    None,  # input_splits for now
                    None,  # output_splits for now
                    ctx.ep_group,
                    False,  # synchronous operation
                )

            (
                grad_output,
                grad_output_c,
                grad_output_t,
                grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx, grad_output, ctx.parallel_mode == "row"
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
                    if ctx.a2a_ag_overlap and ctx.parallel_mode == "row":
                        dgrad = torch.empty(
                            (grad_output.size(0) * ctx.tp_size, weight.size(1)),
                            dtype=ctx.activation_dtype,
                            device=grad_output.device,
                        )
                        _, _ = ring_exchange_overlap_ag(
                            weight=weight,
                            input=grad_output,
                            activation_dtype=ctx.activation_dtype,
                            out=dgrad,
                            ub_algo=None,
                            ub_obj=None,
                            extra_output_tensor=None,
                            ep_aggregate2=False,
                            tp_aggregate2=False,
                            ep_group=ctx.ep_group,
                            ep_size=ctx.ep_size,
                            tp_group=ctx.tp_group,
                            tp_size=ctx.tp_size,
                            layout="NN",
                            grad=True,
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

            # a2a for dgrad RS data from column gemm
            if (
                ctx.parallel_mode == "column"
                and ctx.moe_alltoall_overlap
                and not rs_a2a_overlap
                and not rs_a2a_bulk_overlap
            ):
                if ctx.ep_size != 1:
                    dgrad, _ = alltoall(
                        None,  # don't assign output for now
                        dgrad,
                        None,  # input_splits for now
                        None,  # output_splits for now
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
            None,  # moe_alltoall_overlap
            None,  # moe_ring_exchange
        )
