/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/


#include "../extensions.h"
#include <nvshmem.h>
#include <nvshmemx.h>
#include <nvshmem_api/nvshmem_waitkernel.h>
#include <torch/cuda.h>
#include <cuda.h>
#include <torch/extension.h>
#include <cuda_fp8.h>

namespace nvshmem_api {
 void init_nvshmem_backend(c10d::ProcessGroup *process_group) {
  nvshmemx_init_attr_t attr = {};
  nvshmemx_uniqueid_t id = {};
  printf("enter the function \n");

  int my_rank = process_group->getRank();
  int num_ranks = process_group->getSize();
  if (my_rank == 0) {
       nvshmemx_get_uniqueid(&id);
  }

  auto backend_is_nccl = (process_group->getBackendType() == c10d::ProcessGroup::BackendType::NCCL);

  auto datatensor = torch::from_blob((void*)&id, {static_cast<int64_t>(sizeof(nvshmemx_uniqueid_t) / sizeof(uint8_t))},
                                     at::device(torch::kCPU).dtype(torch::kUInt8));
  auto datatmp = (backend_is_nccl) ? datatensor.cuda() : datatensor;

  c10d::BroadcastOptions bcast_opts;
  bcast_opts.rootRank = 0;
  std::vector<torch::Tensor> datachunk = {datatmp};
  auto work = process_group->broadcast(datachunk, bcast_opts);
  work->wait();

  if (backend_is_nccl) {
    datatensor.copy_(datatmp.cpu());
    datatmp = torch::Tensor();
  }

  printf("Done with broadcast \n");
  nvshmemx_set_attr_uniqueid_args(my_rank, num_ranks, &id, &attr);
  printf("Done with set uniq args \n");
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
  printf("Done with init attr\n");

  int mype = nvshmem_my_pe();
  auto totalnumpe = nvshmem_n_pes();
  printf("my current PE %d, my rank %d and in total %d \n", mype, my_rank, totalnumpe);
  
}

void nvshmem_wait_on_stream(torch::Tensor signal, int wait_kind){
  // use implict streams
  uint64_t* sig_addr = (uint64_t*) signal.data_ptr();
  cudaStream_t cur_stream = (cudaStream_t)at::cuda::getCurrentCUDAStream();

  transformer_engine::nvshmem_wait_on_stream(sig_addr, wait_kind, cur_stream);
}


torch::Tensor create_nvshmem_tensor(const std::vector<int64_t> &shape, c10::ScalarType dtype){
  auto option_gpu =
      at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(c10::cuda::current_device());
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  return at::from_blob(
      nvshmem_malloc(size), shape, [](void *ptr) { nvshmem_free(ptr); }, option_gpu);

}

void nvshmem_send_on_stream(torch::Tensor src, torch::Tensor dst, int peer, torch::Tensor signal){
  void* src_ptr = (void *) src.data_ptr();
  void* dst_ptr = (void *) dst.data_ptr();
  uint64_t* sig_addr = (uint64_t*) signal.data_ptr();
  auto nelement = src.numel() * src.element_size();
  uint64_t sigval = 1;
  at::cuda::CUDAStream cur_stream = at::cuda::getCurrentCUDAStream();

  // NVSHMEM_SIGNAL_ADD NVSHMEM_SIGNAL_SET
  nvshmemx_putmem_signal_on_stream(dst_ptr, src_ptr, nelement, sig_addr, sigval, NVSHMEM_SIGNAL_SET, peer, (cudaStream_t)cur_stream);

}

void nvshmem_send_on_stream_nbi(torch::Tensor src, torch::Tensor dst, int peer, torch::Tensor signal){
  void* src_ptr = (void *) src.data_ptr();
  void* dst_ptr = (void *) dst.data_ptr();
  uint64_t* sig_addr = (uint64_t*) signal.data_ptr();
  auto nelement = src.numel() * src.element_size();
  uint64_t sigval = 1;
  at::cuda::CUDAStream cur_stream = at::cuda::getCurrentCUDAStream();

  // NVSHMEM_SIGNAL_ADD NVSHMEM_SIGNAL_SET
  nvshmemx_putmem_signal_nbi_on_stream(dst_ptr, src_ptr, nelement, sig_addr, sigval, NVSHMEM_SIGNAL_SET, peer, (cudaStream_t)cur_stream);

}
void nvshmem_finalize(){
  nvshmem_finalize();
}
void nvshmem_quiet(){
  nvshmem_quiet();
}
void nvshmem_alltoall_on_stream(torch::Tensor src, torch::Tensor dst){

  void* src_ptr = (void *) src.data_ptr();
  void* dst_ptr = (void *) dst.data_ptr();
  // use global team for now
  nvshmem_team_t myteam = NVSHMEM_TEAM_WORLD;
  auto totalnumpe = nvshmem_n_pes();
  auto src_nelement = src.numel() * src.element_size() ;
  size_t nelement = src_nelement/totalnumpe;
  at::cuda::CUDAStream cur_stream = at::cuda::getCurrentCUDAStream();

  nvshmemx_alltoallmem_on_stream(myteam, dst_ptr, src_ptr, nelement, (cudaStream_t)cur_stream);

}
// void nvshmem_allgather_on_stream_16bit(torch::Tensor src, torch::Tensor dst){
//   float* src_ptr = (float*) src.data_ptr();
//   float* dst_ptr = (float*) dst.data_ptr();
//   // use global team for now
//   nvshmem_team_t myteam = NVSHMEM_TEAM_WORLD;
//   auto totalnumpe = nvshmem_n_pes();
//   auto src_nelement = src.numel();

//   at::cuda::CUDAStream cur_stream = at::cuda::getCurrentCUDAStream();
//   // if( nvshmem_my_pe() == 0){
//   //   printf("src ")
//   // }
//   auto returnval = nvshmemx_float_fcollect_on_stream(myteam, dst_ptr, src_ptr, src_nelement, (cudaStream_t)cur_stream);
//   printf("allgather returned val %d\n", returnval);
// }

void nvshmem_allgather_on_stream_16bit(torch::Tensor src, torch::Tensor dst){
  int16_t* src_ptr = reinterpret_cast<int16_t*> (src.data_ptr());
  int16_t* dst_ptr = reinterpret_cast<int16_t*> (dst.data_ptr());
  // use global team for now
  nvshmem_team_t myteam = NVSHMEM_TEAM_WORLD;
  auto totalnumpe = nvshmem_n_pes();
  auto src_nelement = src.numel();

  at::cuda::CUDAStream cur_stream = at::cuda::getCurrentCUDAStream();
  auto returnval = nvshmemx_int16_fcollect_on_stream(myteam, dst_ptr, src_ptr, src_nelement, (cudaStream_t)cur_stream);
  printf("allgather returned val %d\n", returnval);
}
// void nvshmem alltoall
// nvshmemx_alltoallmem_on_stream(shmem_team_t team, void *dest, const void *source, size_t nelems, cudaStream_t stream)
// int nvshmem_alltoallmem(shmem_team_t team, void *dest, const void *source, size_t nelems)

// nvshmem allgather
// int nvshmemx_TYPENAME_fcollect_on_stream(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems, cudaStream_t stream)


// nvshmem reduce scatter
// seems not support bf16


// nvshmem fence

// nvshmem sync


}