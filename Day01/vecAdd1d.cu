#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>  // Add this header

__global__ void vecAddKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n){
  int i = threadIdx.x+ blockDim.x* blockIdx.x;
  if (i< n){
    C[i] = A[i] + B[i];
  }
}
__host__ torch::Tensor vecAdd(torch::Tensor tensor1, torch::Tensor tensor2){
  TORCH_CHECK(tensor1.is_cuda(), "Tensor must be a cuda tensor");
  TORCH_CHECK(tensor1.is_contiguous(), "Tensor must be contiguous");
  TORCH_CHECK(tensor2.is_cuda(), "Tensor must be a cuda tensor");
  TORCH_CHECK(tensor2.is_contiguous(), "Tensor must be contiguous");

  int size = tensor1.numel();
  float *data1 = tensor1.data_ptr<float>();
  float *data2 = tensor2.data_ptr<float>();
  torch::Tensor result = torch::empty_like(tensor1);
  float *data3 = result.data_ptr<float>();



  int threadsPerBlock = 256;
  int blocksPergrid = (size + threadsPerBlock -1/threadsPerBlock);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  vecAddKernel<<<blocksPergrid,threadsPerBlock,0,stream>>>(data1,data2,data3,size);
  C10_CUDA_CHECK(cudaGetLastError());
  return result;


}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vec_add", &vecAdd, "Add two tensors");
}